"""Calibrated Stage-2 nested CV with warm-started fixed-offset ridge paths.

This version is intentionally separate from the historical uncalibrated and
statsmodels-calibrated implementations.  It parallelizes independent outer
folds with joblib/loky and keeps each inner-fold alpha path sequential so
adjacent ridge fits can share warm starts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

for _thread_env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_thread_env, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from scipy.special import expit
from threadpoolctl import threadpool_limits

HERE = Path(__file__).absolute().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _register_publication_alias(module_name: str, filename: str) -> None:
    """Expose m2_4 sibling files under the import names used by run scripts."""
    if "publication" not in sys.modules:
        package = types.ModuleType("publication")
        package.__path__ = [str(HERE)]
        sys.modules["publication"] = package
    qualified_name = f"publication.{module_name}"
    if qualified_name in sys.modules:
        return
    spec = importlib.util.spec_from_file_location(qualified_name, HERE / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {qualified_name} from {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)


_register_publication_alias(
    "offset_ridge_lbfgs",
    "m2_3_04_optimize_offset_ridge_lbfgs.py",
)
_register_publication_alias(
    "utils",
    "m2_3_14_shared_stage2_utilities.py",
)

from publication.offset_ridge_lbfgs import (  # noqa: E402
    OffsetRidgeFit,
    fit_offset_intercept_lbfgs,
    fit_offset_ridge_lbfgs,
    fit_offset_ridge_path,
    offset_ridge_objective_gradient,
    predict_offset_ridge_lbfgs,
)
from publication.utils import (  # noqa: E402
    CELL_IDS,
    TOX21_TASKS,
    _weighted_binomial_deviance,
    build_task_cell_frame,
    compute_binary_metrics,
    fit_offset_intercept_glm,
    fit_offset_ridge_glm,
    inverse_prevalence_weights,
    load_matched_split,
    predict_offset_ridge_glm,
    standardize_train_test,
    stratified_scaffold_group_kfold,
)

OUTER_K = 5
OUTER_SEED = 2042
OUTER_SEED_MAX_ATTEMPTS = 1000
MIN_CLASS_PER_OUTER_PARTITION = 2
INNER_CV_K = 3
INNER_CV_SEED = 42
ALPHA_LOG10_MIN = -4.0
ALPHA_LOG10_MAX = 8.0
ALPHA_POINTS_PER_DECADE = 2.0
STAGE2_MODEL_SPEC = "frozen_offset_unpenalized_intercept_ridge_biology_lbfgs_v2"

COHORT_DIR = HERE / "_run_output" / "matched_split_rebuilt"
GENE_FEATURES_DIR = HERE / "_run_output" / "gene_features"
CHEM_OFFSET_PATH = HERE / "_run_output" / "chem_offset_k4_deployed" / "chem_offset_final.csv"
DEFAULT_OUT_DIR = HERE / "_run_output" / "nested_cv_k4_calibrated_lbfgs_v2"
VARIANT_SUFFIX = {"standardized": "raw", "residualized": "residualized"}


@dataclass(frozen=True)
class PreparedTaskCell:
    task: str
    cell: str
    variant: str
    X_raw: np.ndarray
    y: np.ndarray
    offset: np.ndarray
    smiles: tuple[str, ...]
    ik: tuple[str, ...]
    gene_names: tuple[str, ...]
    outer_folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    outer_seed_used: int = OUTER_SEED
    # For variant == "residualized" only: the row-aligned unadjusted
    # ("standardized") variant's raw feature matrix for the SAME
    # compounds/rows. When set, standardization for this prepared cell
    # reuses the unadjusted variant's train-fold mean/std instead of its
    # own, so viability-adjusted and unadjusted coefficients are expressed
    # on the same feature scale for a fair before/after comparison. None
    # for variant == "standardized" (no reference needed -- it IS the
    # reference).
    X_raw_scale_reference: np.ndarray | None = None


@dataclass(frozen=True)
class OuterFoldJob:
    prepared: PreparedTaskCell
    fold_id: int
    outer_train_idx: np.ndarray
    outer_test_idx: np.ndarray
    solver: str
    use_sample_weight: bool
    alpha_grid: tuple[float, ...]
    inner_cv_k: int
    inner_cv_seed: int
    maxiter: int
    retry_maxiter: int
    gtol: float
    acceptance_gradient: float
    ftol: float


@dataclass
class OuterFoldOutcome:
    task: str
    cell: str
    variant: str
    fold_id: int
    success: bool
    error: str
    fold_row: dict[str, Any] | None
    oof_rows: list[dict[str, Any]]
    beta: np.ndarray | None
    gene_names: tuple[str, ...]
    alpha_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ReferenceFit:
    params: np.ndarray
    alpha: float
    success: bool
    message: str
    n_iterations: int
    n_function_evaluations: int
    gradient_inf_norm: float
    objective: float
    elapsed_seconds: float
    warm_started: bool
    retry_used: bool
    sample_weighted: bool

    @property
    def intercept(self) -> float:
        return float(self.params[0])

    @property
    def beta(self) -> np.ndarray:
        return np.asarray(self.params[1:], dtype=float)


def alpha_grid(points_per_decade: float = ALPHA_POINTS_PER_DECADE) -> tuple[float, ...]:
    n_points = max(
        2,
        round((ALPHA_LOG10_MAX - ALPHA_LOG10_MIN) * points_per_decade) + 1,
    )
    return tuple(
        float(value)
        for value in np.power(
            10.0,
            np.linspace(ALPHA_LOG10_MAX, ALPHA_LOG10_MIN, n_points),
        )
    )


@lru_cache(maxsize=None)
def _load_cell_data(
    cell: str,
    variant: str,
    cohort_dir: str,
    gene_features_dir: str,
):
    """Load each cell/variant table once while preparing all task-fold jobs."""
    compounds = load_matched_split(
        str(
            Path(cohort_dir)
            / f"tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl"
        )
    )
    suffix = VARIANT_SUFFIX[variant]
    gene_features = pd.read_csv(
        Path(gene_features_dir) / f"{cell}_gene_features_{suffix}.csv"
    )
    return compounds, gene_features


def load_chem_offsets(path: Path = CHEM_OFFSET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def validate_chem_offsets(
    offsets: pd.DataFrame,
    tasks: list[str],
) -> pd.DataFrame:
    required = ["ik"] + [f"{task}_chem_logit" for task in tasks]
    missing_columns = [column for column in required if column not in offsets]
    if missing_columns:
        raise ValueError(
            "Chemical-offset table is missing columns: "
            + ", ".join(missing_columns)
        )
    if offsets["ik"].isna().any():
        raise ValueError("Chemical-offset table contains missing ik values")
    duplicated = offsets.loc[offsets["ik"].duplicated(keep=False), "ik"]
    if not duplicated.empty:
        examples = ", ".join(duplicated.astype(str).drop_duplicates().head(5))
        raise ValueError(
            "Chemical-offset table must contain one row per ik; duplicates: "
            + examples
        )
    for task in tasks:
        column = f"{task}_chem_logit"
        values = offsets[column].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Chemical-offset column contains non-finite values: {column}")
    return offsets


def find_evaluable_outer_folds(
    smiles: tuple[str, ...],
    y: np.ndarray,
    outer_k: int = OUTER_K,
    initial_seed: int = OUTER_SEED,
    max_attempts: int = OUTER_SEED_MAX_ATTEMPTS,
    min_class_count: int = MIN_CLASS_PER_OUTER_PARTITION,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], int]:
    """Return the first deterministic scaffold split with evaluable folds."""
    for seed in range(initial_seed, initial_seed + max_attempts):
        folds = stratified_scaffold_group_kfold(
            smiles,
            y,
            k=outer_k,
            seed=seed,
        )
        valid_folds = tuple(
            (np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int))
            for train_idx, test_idx in folds
            if len(train_idx) > 0 and len(test_idx) > 0
        )
        if len(valid_folds) != outer_k:
            continue
        if all(
            np.sum(y[indices] == class_value) >= min_class_count
            for train_idx, test_idx in valid_folds
            for indices in (train_idx, test_idx)
            for class_value in (0.0, 1.0)
        ):
            return valid_folds, seed
    raise ValueError(
        f"no evaluable {outer_k}-fold scaffold split found across "
        f"{max_attempts} deterministic seeds beginning at {initial_seed}"
    )


def prepare_task_cell(
    task: str,
    cell: str,
    variant: str,
    offsets: pd.DataFrame,
    outer_k: int = OUTER_K,
    outer_seed: int = OUTER_SEED,
    cohort_dir: Path = COHORT_DIR,
    gene_features_dir: Path = GENE_FEATURES_DIR,
) -> PreparedTaskCell:
    compounds, gene_features = _load_cell_data(
        cell,
        variant,
        str(Path(cohort_dir).resolve()),
        str(Path(gene_features_dir).resolve()),
    )
    df, gene_names = build_task_cell_frame(compounds, gene_features, task)
    if df.empty:
        raise ValueError("matched biological cohort is empty")
    row_count = len(df)
    offset_column = f"{task}_chem_logit"
    df = df.merge(
        offsets[["ik", offset_column]],
        on="ik",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if len(df) != row_count:
        raise RuntimeError("Chemical-offset merge changed the cohort row count")
    missing_offset = df["_merge"] != "both"
    if missing_offset.any():
        examples = ", ".join(df.loc[missing_offset, "ik"].astype(str).head(5))
        raise ValueError(
            f"{int(missing_offset.sum())} cohort rows lack a chemical offset; "
            f"examples: {examples}"
        )
    df = df.drop(columns="_merge")
    if not np.all(np.isfinite(df[offset_column].to_numpy(dtype=np.float64))):
        raise ValueError("Merged chemical offsets contain non-finite values")
    if df[task].nunique() != 2:
        raise ValueError("task-cell cohort is not binary-evaluable")

    y = df[task].to_numpy(dtype=np.float64)
    smiles = tuple(df["smiles_canon"].astype(str))
    valid_folds, outer_seed_used = find_evaluable_outer_folds(
        smiles,
        y,
        outer_k=outer_k,
        initial_seed=outer_seed,
    )

    scale_reference = None
    if variant == "residualized":
        ref_compounds, ref_gene_features = _load_cell_data(
            cell,
            "standardized",
            str(Path(cohort_dir).resolve()),
            str(Path(gene_features_dir).resolve()),
        )
        ref_df, ref_gene_names = build_task_cell_frame(
            ref_compounds, ref_gene_features, task
        )
        if tuple(ref_gene_names) != tuple(gene_names):
            raise RuntimeError(
                "Unadjusted scale-reference gene order does not match the "
                f"residualized variant for {task}/{cell}"
            )
        ref_ik = tuple(ref_df["ik"].astype(str))
        if ref_ik != tuple(df["ik"].astype(str)):
            raise RuntimeError(
                "Unadjusted scale-reference compound order does not match "
                f"the residualized variant for {task}/{cell}; cannot borrow "
                "its standardization statistics"
            )
        scale_reference = np.ascontiguousarray(
            ref_df[ref_gene_names].to_numpy(dtype=np.float64)
        )

    return PreparedTaskCell(
        task=task,
        cell=cell,
        variant=variant,
        X_raw=np.ascontiguousarray(df[gene_names].to_numpy(dtype=np.float64)),
        y=y,
        offset=df[f"{task}_chem_logit"].to_numpy(dtype=np.float64),
        smiles=smiles,
        ik=tuple(df["ik"].astype(str)),
        gene_names=tuple(gene_names),
        outer_folds=valid_folds,
        outer_seed_used=outer_seed_used,
        X_raw_scale_reference=scale_reference,
    )


def _statsmodels_fit(
    X,
    y,
    offset,
    alpha,
    sample_weight,
    maxiter,
    acceptance_gradient,
) -> ReferenceFit:
    started = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=UserWarning)
        result = fit_offset_ridge_glm(
            X,
            y,
            offset,
            alpha,
            L1_wt=0.0,
            include_intercept=True,
            var_weights=sample_weight,
            maxiter=maxiter,
            cnvrg_tol=acceptance_gradient,
        )
    warned = any("ridge optimization may have failed" in str(item.message) for item in caught)
    params = np.asarray(result.params, dtype=float)
    objective, gradient = offset_ridge_objective_gradient(
        params,
        X,
        y,
        offset,
        alpha,
        sample_weight=sample_weight,
    )
    gradient_inf_norm = float(np.linalg.norm(gradient, ord=np.inf))
    finite = bool(
        np.all(np.isfinite(params))
        and np.isfinite(objective)
        and np.isfinite(gradient_inf_norm)
    )
    success = bool(
        not warned
        and finite
        and gradient_inf_norm <= acceptance_gradient
    )
    if warned:
        message = "statsmodels convergence warning"
    elif not finite:
        message = "statsmodels returned non-finite diagnostics"
    elif gradient_inf_norm > acceptance_gradient:
        message = (
            "statsmodels gradient gate failed: "
            f"{gradient_inf_norm:.3e} > {acceptance_gradient:.3e}"
        )
    else:
        message = "statsmodels completed"
    return ReferenceFit(
        params=params,
        alpha=float(alpha),
        success=success,
        message=message,
        n_iterations=-1,
        n_function_evaluations=-1,
        gradient_inf_norm=gradient_inf_norm,
        objective=float(objective),
        elapsed_seconds=perf_counter() - started,
        warm_started=False,
        retry_used=False,
        sample_weighted=sample_weight is not None,
    )


def _fit_path(
    solver: str,
    X,
    y,
    offset,
    alphas,
    sample_weight,
    maxiter,
    retry_maxiter,
    gtol,
    acceptance_gradient,
    ftol,
):
    if solver == "lbfgs":
        return fit_offset_ridge_path(
            X,
            y,
            offset,
            alphas,
            sample_weight=sample_weight,
            maxiter=maxiter,
            retry_maxiter=retry_maxiter,
            gtol=gtol,
            acceptance_gradient=acceptance_gradient,
            ftol=ftol,
        )
    return [
        _statsmodels_fit(
            X,
            y,
            offset,
            alpha,
            sample_weight,
            maxiter,
            acceptance_gradient,
        )
        for alpha in sorted(alphas, reverse=True)
    ]


def _predict(fitted, X, offset, solver: str):
    if solver == "lbfgs":
        return predict_offset_ridge_lbfgs(fitted, X, offset)[0]
    return predict_offset_ridge_glm(fitted, X, offset, include_intercept=True)


def _standardize_stage2_features(
    variant: str,
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    scale_reference_train_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if variant not in VARIANT_SUFFIX:
        raise ValueError(f"unknown Stage-2 feature variant: {variant}")
    if scale_reference_train_raw is None:
        return standardize_train_test(X_train_raw, X_test_raw)
    # Fair before/after comparison: standardize the viability-adjusted
    # (residualized) features using the SAME train-fold mean/std as the
    # unadjusted counterpart, rather than the residualized features' own
    # (generally smaller, since a shared axis was removed) variance. This
    # keeps ridge coefficients on both variants on the same feature scale.
    mu = scale_reference_train_raw.mean(axis=0)
    sd = scale_reference_train_raw.std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train_raw - mu) / sd, (X_test_raw - mu) / sd


def _select_alpha_from_losses(
    losses_by_alpha: dict[float, list[float]],
) -> dict[str, Any] | None:
    if not losses_by_alpha:
        return None
    mean_loss = {
        alpha: float(np.mean(losses))
        for alpha, losses in losses_by_alpha.items()
        if losses
    }
    if not mean_loss:
        return None

    alpha_min = min(mean_loss, key=mean_loss.get)
    losses_at_min = losses_by_alpha[alpha_min]
    se_min = (
        float(np.std(losses_at_min, ddof=1) / np.sqrt(len(losses_at_min)))
        if len(losses_at_min) > 1
        else 0.0
    )
    threshold = mean_loss[alpha_min] + se_min
    ascending = sorted(mean_loss)
    minimum_position = ascending.index(alpha_min)
    selected_position = minimum_position
    for position in range(minimum_position + 1, len(ascending)):
        if mean_loss[ascending[position]] <= threshold:
            selected_position = position
        else:
            break
    selected = ascending[selected_position]

    log_grid = np.log10(np.asarray(ascending, dtype=float))
    step = float(np.min(np.diff(log_grid))) if len(log_grid) > 1 else 0.0
    lower, upper = float(log_grid.min()), float(log_grid.max())
    selected_log = float(np.log10(selected))
    boundary = bool(
        selected_log <= lower + step / 2.0
        or selected_log >= upper - step / 2.0
    )

    flat_epsilon = max(1e-6, se_min * 0.01)
    larger = [alpha for alpha in ascending if alpha > alpha_min]
    saturation_start = None
    for end in range(2, len(larger)):
        tail = larger[end - 2 : end + 1]
        changes = [
            abs(mean_loss[tail[index]] - mean_loss[tail[index - 1]])
            for index in range(1, len(tail))
        ]
        if all(change < flat_epsilon for change in changes):
            saturation_start = tail[0]
            break
    selected_in_saturated_region = bool(
        saturation_start is not None and selected >= saturation_start
    )

    return {
        "alpha_star": float(selected),
        "alpha_min_loss": float(alpha_min),
        "cv_loss_at_alpha_min": float(mean_loss[alpha_min]),
        "cv_loss_se_at_alpha_min": se_min,
        "alpha_boundary_hit": boundary,
        "beta_saturated_no_bio_signal": selected_in_saturated_region,
        "saturation_start_alpha": (
            float(saturation_start) if saturation_start is not None else float("nan")
        ),
    }


def _find_binary_evaluable_inner_folds(
    *,
    smiles: list[str],
    y: np.ndarray,
    k: int,
    requested_seed: int,
    max_attempts: int = 1000,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Find the first deterministic inner split with both classes per fold."""
    for seed in range(requested_seed, requested_seed + max_attempts):
        folds = stratified_scaffold_group_kfold(smiles, y, k=k, seed=seed)
        if len(folds) != k:
            continue
        if all(
            np.unique(y[np.asarray(train_idx, dtype=int)]).size == 2
            and np.unique(y[np.asarray(valid_idx, dtype=int)]).size == 2
            for train_idx, valid_idx in folds
        ):
            return folds, seed
    raise ValueError(
        "No binary-evaluable inner scaffold split found in seed window "
        f"[{requested_seed}, {requested_seed + max_attempts - 1}]"
    )


def run_outer_fold(job: OuterFoldJob) -> OuterFoldOutcome:
    prepared = job.prepared
    task, cell, variant = prepared.task, prepared.cell, prepared.variant
    empty = OuterFoldOutcome(
        task=task,
        cell=cell,
        variant=variant,
        fold_id=job.fold_id,
        success=False,
        error="",
        fold_row=None,
        oof_rows=[],
        beta=None,
        gene_names=prepared.gene_names,
        alpha_rows=[],
    )
    try:
        with threadpool_limits(limits=1):
            train_idx = job.outer_train_idx
            test_idx = job.outer_test_idx
            y_train, y_test = prepared.y[train_idx], prepared.y[test_idx]
            if (
                np.sum(y_train == 1) < 2
                or np.sum(y_train == 0) < 2
                or np.sum(y_test == 1) < 2
                or np.sum(y_test == 0) < 2
            ):
                empty.error = "outer fold has fewer than two observations in one class"
                return empty

            X_train_raw = prepared.X_raw[train_idx]
            X_test_raw = prepared.X_raw[test_idx]
            scale_ref_train = (
                prepared.X_raw_scale_reference[train_idx]
                if prepared.X_raw_scale_reference is not None
                else None
            )
            X_train, X_test = _standardize_stage2_features(
                variant, X_train_raw, X_test_raw, scale_ref_train
            )
            offset_train, offset_test = prepared.offset[train_idx], prepared.offset[test_idx]
            smiles_train = [prepared.smiles[index] for index in train_idx]

            inner_folds, inner_seed_used = _find_binary_evaluable_inner_folds(
                smiles=smiles_train,
                y=y_train,
                k=job.inner_cv_k,
                requested_seed=job.inner_cv_seed,
            )
            inner_seed_metadata = {
                "inner_cv_seed_requested": int(job.inner_cv_seed),
                "inner_cv_seed_used": int(inner_seed_used),
                "inner_cv_seed_search_attempts": int(
                    inner_seed_used - job.inner_cv_seed + 1
                ),
            }

            losses_by_alpha = {alpha: [] for alpha in job.alpha_grid}
            alpha_rows = []
            for inner_fold_id, (inner_train_idx, inner_valid_idx) in enumerate(inner_folds):
                y_inner_train = y_train[inner_train_idx]
                y_inner_valid = y_train[inner_valid_idx]
                if np.unique(y_inner_train).size != 2 or np.unique(y_inner_valid).size != 2:
                    empty.error = f"inner fold {inner_fold_id} is not binary-evaluable"
                    return empty

                X_inner_train_raw = X_train_raw[inner_train_idx]
                X_inner_valid_raw = X_train_raw[inner_valid_idx]
                scale_ref_inner_train = (
                    scale_ref_train[inner_train_idx]
                    if scale_ref_train is not None
                    else None
                )
                X_inner_train, X_inner_valid = _standardize_stage2_features(
                    variant, X_inner_train_raw, X_inner_valid_raw, scale_ref_inner_train
                )
                inner_weight = (
                    inverse_prevalence_weights(y_inner_train)
                    if job.use_sample_weight
                    else None
                )
                validation_weight = (
                    inverse_prevalence_weights(y_inner_valid)
                    if job.use_sample_weight
                    else None
                )
                fitted_path = _fit_path(
                    job.solver,
                    X_inner_train,
                    y_inner_train,
                    offset_train[inner_train_idx],
                    job.alpha_grid,
                    inner_weight,
                    job.maxiter,
                    job.retry_maxiter,
                    job.gtol,
                    job.acceptance_gradient,
                    job.ftol,
                )
                by_alpha = {fit.alpha: fit for fit in fitted_path}
                for alpha in job.alpha_grid:
                    fit = by_alpha[alpha]
                    loss = float("nan")
                    if fit.success:
                        probability = _predict(
                            fit,
                            X_inner_valid,
                            offset_train[inner_valid_idx],
                            job.solver,
                        )
                        loss = _weighted_binomial_deviance(
                            y_inner_valid,
                            probability,
                            validation_weight,
                        )
                        losses_by_alpha[alpha].append(loss)
                    alpha_rows.append(
                        {
                            "task": task,
                            "cell": cell,
                            "variant": variant,
                            "outer_fold": job.fold_id,
                            "outer_split_seed": prepared.outer_seed_used,
                            "inner_fold": inner_fold_id,
                            "solver": job.solver,
                            "alpha": alpha,
                            "success": fit.success,
                            "validation_deviance": loss,
                            "warm_started": fit.warm_started,
                            "retry_used": fit.retry_used,
                            "n_iterations": fit.n_iterations,
                            "n_function_evaluations": fit.n_function_evaluations,
                            **inner_seed_metadata,
                            "gradient_inf_norm": fit.gradient_inf_norm,
                            "objective": fit.objective,
                            "elapsed_seconds": fit.elapsed_seconds,
                            "message": fit.message,
                        }
                    )

            complete_losses = {
                alpha: losses
                for alpha, losses in losses_by_alpha.items()
                if len(losses) == job.inner_cv_k
            }
            alpha_info = _select_alpha_from_losses(complete_losses)
            if alpha_info is None:
                empty.error = "no alpha converged in every inner fold"
                empty.alpha_rows = alpha_rows
                return empty

            selected_alpha = alpha_info["alpha_star"]
            train_weight = (
                inverse_prevalence_weights(y_train)
                if job.use_sample_weight
                else None
            )
            if job.solver == "lbfgs":
                null_fit = fit_offset_intercept_lbfgs(
                    y_train, offset_train, sample_weight=train_weight
                )
                final_fit = fit_offset_ridge_lbfgs(
                    X_train,
                    y_train,
                    offset_train,
                    selected_alpha,
                    sample_weight=train_weight,
                    maxiter=job.maxiter,
                    retry_maxiter=job.retry_maxiter,
                    gtol=job.gtol,
                    acceptance_gradient=job.acceptance_gradient,
                    ftol=job.ftol,
                )
                p_bio_test = predict_offset_ridge_lbfgs(
                    final_fit, X_test, offset_test
                )[0]
                null_intercept = null_fit.intercept
            else:
                null_result = fit_offset_intercept_glm(
                    y_train, offset_train, var_weights=train_weight
                )
                null_intercept = float(np.asarray(null_result.params).reshape(-1)[0])
                final_fit = _statsmodels_fit(
                    X_train,
                    y_train,
                    offset_train,
                    selected_alpha,
                    train_weight,
                    job.maxiter,
                    job.acceptance_gradient,
                )
                p_bio_test = _predict(
                    final_fit, X_test, offset_test, job.solver
                )
            if not final_fit.success:
                empty.error = f"final selected-alpha fit failed: {final_fit.message}"
                empty.alpha_rows = alpha_rows
                return empty

            p_stage1_test = expit(offset_test)
            p_chem_test = expit(offset_test + null_intercept)
            m_stage1 = compute_binary_metrics(y_test, p_stage1_test)
            m_chem = compute_binary_metrics(y_test, p_chem_test)
            m_bio = compute_binary_metrics(y_test, p_bio_test)

            fold_row = {
                "task": task,
                "cell": cell,
                "variant": variant,
                "fold_id": job.fold_id,
                "outer_split_seed": prepared.outer_seed_used,
                **inner_seed_metadata,
                "solver": job.solver,
                "chemical_offset_frozen": True,
                "chemical_offset_coefficient": 1.0,
                "stage2_model_spec": STAGE2_MODEL_SPEC,
                "intercept_penalized": False,
                "use_sample_weight": job.use_sample_weight,
                "stage2_likelihood": (
                    "weighted_sensitivity"
                    if job.use_sample_weight
                    else "unweighted_publication_primary"
                ),
                "null_intercept": null_intercept,
                "bio_intercept": final_fit.intercept,
                "beta_l2_norm": float(np.linalg.norm(final_fit.beta)),
                "alpha_star_outer": selected_alpha,
                "alpha_min_loss": alpha_info["alpha_min_loss"],
                "alpha_boundary_hit": alpha_info["alpha_boundary_hit"],
                "beta_saturated_no_bio_signal": alpha_info[
                    "beta_saturated_no_bio_signal"
                ],
                "saturation_start_alpha": alpha_info["saturation_start_alpha"],
                "n_outer_train": len(train_idx),
                "n_outer_test": len(test_idx),
                "auprc_stage1_raw": m_stage1["auprc"],
                "auprc_chem": m_chem["auprc"],
                "auprc_bio": m_bio["auprc"],
                "delta_auprc": m_bio["auprc"] - m_chem["auprc"],
                "auroc_stage1_raw": m_stage1["auroc"],
                "auroc_chem": m_chem["auroc"],
                "auroc_bio": m_bio["auroc"],
                "delta_auroc": m_bio["auroc"] - m_chem["auroc"],
                "final_success": final_fit.success,
                "final_retry_used": final_fit.retry_used,
                "final_n_iterations": final_fit.n_iterations,
                "final_n_function_evaluations": final_fit.n_function_evaluations,
                "final_gradient_inf_norm": final_fit.gradient_inf_norm,
                "final_objective": final_fit.objective,
                "final_elapsed_seconds": final_fit.elapsed_seconds,
            }
            oof_rows = [
                {
                    "task": task,
                    "cell": cell,
                    "variant": variant,
                    "fold_id": job.fold_id,
                    "outer_split_seed": prepared.outer_seed_used,
                    **inner_seed_metadata,
                    "ik": prepared.ik[index],
                    "stage2_model_spec": STAGE2_MODEL_SPEC,
                    "solver": job.solver,
                    "y": float(prepared.y[index]),
                    "p_chem_stage1_raw": float(p_stage1_test[position]),
                    "p_chem_only": float(p_chem_test[position]),
                    "p_chem_bio": float(p_bio_test[position]),
                    "null_intercept": null_intercept,
                    "bio_intercept": final_fit.intercept,
                }
                for position, index in enumerate(test_idx)
            ]
            return OuterFoldOutcome(
                task=task,
                cell=cell,
                variant=variant,
                fold_id=job.fold_id,
                success=True,
                error="",
                fold_row=fold_row,
                oof_rows=oof_rows,
                beta=np.asarray(final_fit.beta, dtype=float),
                gene_names=prepared.gene_names,
                alpha_rows=alpha_rows,
            )
    except Exception as exc:
        empty.error = f"{type(exc).__name__}: {exc}"
        return empty


def _write_group_outputs(
    outcomes: list[OuterFoldOutcome],
    output_dir: Path,
    variant: str,
    task: str,
    cell: str,
) -> dict[str, Any] | None:
    group = sorted(
        [
            outcome
            for outcome in outcomes
            if outcome.variant == variant
            and outcome.task == task
            and outcome.cell == cell
        ],
        key=lambda outcome: outcome.fold_id,
    )
    successful = [outcome for outcome in group if outcome.success]
    if not successful:
        return None

    fold_df = pd.DataFrame([outcome.fold_row for outcome in successful])
    oof_df = pd.DataFrame(
        [row for outcome in successful for row in outcome.oof_rows]
    )
    beta_rows = []
    for outcome in successful:
        for gene, beta in zip(outcome.gene_names, outcome.beta):
            beta_rows.append(
                {
                    "task": task,
                    "cell": cell,
                    "variant": variant,
                    "fold_id": outcome.fold_id,
                    "outer_split_seed": outcome.fold_row["outer_split_seed"],
                    "gene": gene,
                    "beta": float(beta),
                    "alpha_star_outer": outcome.fold_row["alpha_star_outer"],
                    "bio_intercept": outcome.fold_row["bio_intercept"],
                    "beta_saturated_no_bio_signal": outcome.fold_row[
                        "beta_saturated_no_bio_signal"
                    ],
                }
            )
    alpha_df = pd.DataFrame(
        [row for outcome in group for row in outcome.alpha_rows]
    )

    stem = f"{variant}_{task}__{cell}"
    fold_df.to_csv(output_dir / f"nested_cv_{stem}.csv", index=False)
    oof_df.to_csv(
        output_dir / f"nested_cv_oof_predictions_{stem}.csv", index=False
    )
    pd.DataFrame(beta_rows).to_csv(
        output_dir / f"nested_cv_beta_{stem}.csv", index=False
    )
    if not alpha_df.empty:
        alpha_df.to_csv(
            output_dir / f"nested_cv_alpha_diagnostics_{stem}.csv", index=False
        )

    delta = fold_df["delta_auprc"].to_numpy(dtype=float)
    summary = {
        "task": task,
        "cell": cell,
        "variant": variant,
        "solver": str(fold_df["solver"].iloc[0]),
        "outer_split_seed": int(fold_df["outer_split_seed"].iloc[0]),
        "n_outer_folds": len(fold_df),
        "n_outer_folds_failed": len(group) - len(successful),
        "mean_delta_auprc": float(delta.mean()),
        "std_delta_auprc": float(delta.std(ddof=1)) if len(delta) > 1 else 0.0,
        "min_delta_auprc": float(delta.min()),
        "max_delta_auprc": float(delta.max()),
        "n_alpha_boundary_hits": int(fold_df["alpha_boundary_hit"].sum()),
        "n_folds_saturated_no_bio_signal": int(
            fold_df["beta_saturated_no_bio_signal"].sum()
        ),
        "all_folds_saturated_no_bio_signal": bool(
            fold_df["beta_saturated_no_bio_signal"].all()
        ),
    }
    pd.DataFrame([summary]).to_csv(
        output_dir / f"nested_cv_summary_{stem}.csv", index=False
    )
    return summary


def run_analysis(
    output_dir: Path,
    variants: list[str],
    tasks: list[str],
    cells: list[str],
    chem_offset_path: Path = CHEM_OFFSET_PATH,
    cohort_dir: Path = COHORT_DIR,
    gene_features_dir: Path = GENE_FEATURES_DIR,
    solver: str = "lbfgs",
    n_jobs: int = -1,
    points_per_decade: float = ALPHA_POINTS_PER_DECADE,
    use_sample_weight: bool = False,
    maxiter: int = 500,
    retry_maxiter: int = 2000,
    gtol: float = 1e-6,
    acceptance_gradient: float = 1e-5,
    ftol: float = 1e-14,
) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite or resume existing output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    offsets = validate_chem_offsets(load_chem_offsets(chem_offset_path), tasks)
    alphas = alpha_grid(points_per_decade)

    requested_groups = [
        (variant, task, cell)
        for variant in variants
        for cell in cells
        for task in tasks
    ]
    prepared_sets = []
    preparation_failure_rows = []
    for variant in variants:
        for cell in cells:
            for task in tasks:
                try:
                    prepared = prepare_task_cell(
                        task,
                        cell,
                        variant,
                        offsets,
                        cohort_dir=cohort_dir,
                        gene_features_dir=gene_features_dir,
                    )
                    prepared_sets.append(prepared)
                except Exception as exc:
                    error = f"preparation failed: {type(exc).__name__}: {exc}"
                    preparation_failure_rows.extend(
                        {
                            "task": task,
                            "cell": cell,
                            "variant": variant,
                            "fold_id": fold_id,
                            "error": error,
                        }
                        for fold_id in range(OUTER_K)
                    )

    jobs = [
        OuterFoldJob(
            prepared=prepared,
            fold_id=fold_id,
            outer_train_idx=train_idx,
            outer_test_idx=test_idx,
            solver=solver,
            use_sample_weight=use_sample_weight,
            alpha_grid=alphas,
            inner_cv_k=INNER_CV_K,
            inner_cv_seed=INNER_CV_SEED,
            maxiter=maxiter,
            retry_maxiter=retry_maxiter,
            gtol=gtol,
            acceptance_gradient=acceptance_gradient,
            ftol=ftol,
        )
        for prepared in prepared_sets
        for fold_id, (train_idx, test_idx) in enumerate(prepared.outer_folds)
    ]
    expected_outer_fold_jobs = len(requested_groups) * OUTER_K
    resolved_jobs = min(max(1, effective_n_jobs(n_jobs)), max(1, len(jobs)))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "calibrated_stage2_lbfgs_v2",
        "stage2_model_spec": STAGE2_MODEL_SPEC,
        "solver": solver,
        "chemical_offset_path": str(chem_offset_path.resolve()),
        "chemical_offset_frozen": True,
        "chemical_offset_coefficient": 1.0,
        "intercept_penalized": False,
        "stage2_likelihood": (
            "weighted_sensitivity"
            if use_sample_weight
            else "unweighted_publication_primary"
        ),
        "variants": variants,
        "tasks": tasks,
        "cells": cells,
        "outer_folds": OUTER_K,
        "outer_seed_initial": OUTER_SEED,
        "outer_seed_max_attempts": OUTER_SEED_MAX_ATTEMPTS,
        "outer_min_class_per_partition": MIN_CLASS_PER_OUTER_PARTITION,
        "outer_split_policy": (
            "first deterministic stratified scaffold split with at least "
            "two observations per class in every train and test partition"
        ),
        "inner_folds": INNER_CV_K,
        "alpha_grid": list(alphas),
        "requested_n_jobs": n_jobs,
        "resolved_n_jobs": resolved_jobs,
        "available_logical_cpus": os.cpu_count(),
        "parallel_unit": "outer_fold",
        "backend": "loky",
        "inner_max_num_threads": 1,
        "n_requested_groups": len(requested_groups),
        "n_expected_outer_fold_jobs": expected_outer_fold_jobs,
        "n_outer_fold_jobs": len(jobs),
        "n_launched_outer_fold_jobs": len(jobs),
        "n_preparation_failed_outer_folds": len(preparation_failure_rows),
        "analysis_started": True,
        "full_matrix_requested": bool(
            len(variants) == 2
            and set(tasks) == set(TOX21_TASKS)
            and set(cells) == set(CELL_IDS)
        ),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"[calibrated-v2] {len(jobs)}/{expected_outer_fold_jobs} "
        "outer-fold jobs prepared; "
        f"joblib workers={resolved_jobs}; solver={solver}"
    )
    with parallel_config(
        backend="loky",
        n_jobs=resolved_jobs,
        inner_max_num_threads=1,
    ):
        outcomes = Parallel(
            n_jobs=resolved_jobs,
            batch_size=1,
            pre_dispatch=resolved_jobs,
            max_nbytes="10M",
            mmap_mode="r",
        )(delayed(run_outer_fold)(job) for job in jobs)

    failure_rows = list(preparation_failure_rows)
    failure_rows.extend(
        {
            "task": outcome.task,
            "cell": outcome.cell,
            "variant": outcome.variant,
            "fold_id": outcome.fold_id,
            "error": outcome.error,
        }
        for outcome in outcomes
        if not outcome.success
    )

    summaries = []
    for variant in variants:
        for cell in cells:
            for task in tasks:
                summary = _write_group_outputs(
                    outcomes, output_dir, variant, task, cell
                )
                if summary is not None:
                    summaries.append(summary)
    summary_by_group = {
        (summary["variant"], summary["task"], summary["cell"]): summary
        for summary in summaries
    }
    failed_group_folds = {
        (row["variant"], row["task"], row["cell"], int(row["fold_id"]))
        for row in failure_rows
    }
    successful_group_folds = {
        (outcome.variant, outcome.task, outcome.cell, int(outcome.fold_id))
        for outcome in outcomes
        if outcome.success
    }
    for variant, task, cell in requested_groups:
        summary = summary_by_group.get((variant, task, cell))
        successful_fold_count = (
            int(summary["n_outer_folds"]) if summary is not None else 0
        )
        if successful_fold_count == OUTER_K:
            continue
        for fold_id in range(OUTER_K):
            key = (variant, task, cell, fold_id)
            if key in failed_group_folds or key in successful_group_folds:
                continue
            failure_rows.append(
                {
                    "task": task,
                    "cell": cell,
                    "variant": variant,
                    "fold_id": fold_id,
                    "error": (
                        "completeness check failed: requested group has "
                        f"{successful_fold_count}/{OUTER_K} successful outer folds"
                    ),
                }
            )
            failed_group_folds.add(key)

    if failure_rows:
        pd.DataFrame(failure_rows).to_csv(
            output_dir / "outer_fold_failures.csv", index=False
        )
    pd.DataFrame(summaries).to_csv(
        output_dir / "nested_cv_summary_all.csv", index=False
    )
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_successful_outer_folds"] = int(
        sum(outcome.success for outcome in outcomes)
    )
    manifest["n_failed_outer_folds"] = len(failed_group_folds)
    manifest["n_completed_groups"] = int(
        sum(
            int(summary["n_outer_folds"]) == OUTER_K
            for summary in summaries
        )
    )
    complete = bool(
        not failure_rows
        and len(outcomes) == expected_outer_fold_jobs
        and len(summaries) == len(requested_groups)
        and manifest["n_successful_outer_folds"] == expected_outer_fold_jobs
        and manifest["n_completed_groups"] == len(requested_groups)
    )
    manifest["analysis_complete"] = complete
    completion_name = "RUN_COMPLETE.json" if complete else "RUN_INCOMPLETE.json"
    (output_dir / completion_name).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not complete:
        raise RuntimeError(
            f"{len(failed_group_folds)} requested outer-fold jobs failed or "
            "were missing; "
            f"see {output_dir / 'outer_fold_failures.csv'}"
        )
    return summaries


def _parse_csv_list(value: str | None, allowed: list[str]) -> list[str]:
    if value is None:
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run versioned calibrated Stage-2 outer-fold jobs."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["standardized", "residualized", "both"], default="both")
    parser.add_argument("--tasks", help="Comma-separated task subset")
    parser.add_argument("--cells", help="Comma-separated cell subset")
    parser.add_argument("--chem-offset-path", type=Path, default=CHEM_OFFSET_PATH)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    parser.add_argument(
        "--gene-features-dir",
        type=Path,
        default=GENE_FEATURES_DIR,
    )
    parser.add_argument("--ridge-solver", choices=["lbfgs", "statsmodels"], default="lbfgs")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--points-per-decade", type=float, default=ALPHA_POINTS_PER_DECADE)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--retry-maxiter", type=int, default=2000)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--acceptance-gradient", type=float, default=1e-5)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--use-sample-weight", action="store_true")
    args = parser.parse_args()

    variants = (
        ["standardized", "residualized"]
        if args.variant == "both"
        else [args.variant]
    )
    run_analysis(
        output_dir=args.out_dir,
        variants=variants,
        tasks=_parse_csv_list(args.tasks, TOX21_TASKS),
        cells=_parse_csv_list(args.cells, CELL_IDS),
        chem_offset_path=args.chem_offset_path,
        cohort_dir=args.cohort_dir,
        gene_features_dir=args.gene_features_dir,
        solver=args.ridge_solver,
        n_jobs=args.n_jobs,
        points_per_decade=args.points_per_decade,
        use_sample_weight=args.use_sample_weight,
        maxiter=args.maxiter,
        retry_maxiter=args.retry_maxiter,
        gtol=args.gtol,
        acceptance_gradient=args.acceptance_gradient,
        ftol=args.ftol,
    )


if __name__ == "__main__":
    main()
