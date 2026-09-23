"""Stage-2 nested CV with calibration fitted first and frozen during ridge fitting.

Each training partition estimates its offset-only calibration intercept.
The intercept then joins the frozen chemical logit as a fixed offset for
beta-only ridge fitting. Independent outer folds run with joblib/loky;
each inner-fold alpha path shares warm starts between adjacent ridge fits.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
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

HERE = Path(__file__).resolve().parent
_STANDALONE_ROOT = HERE.parent  # pipeline root; retained name for config compatibility
if str(_STANDALONE_ROOT) not in sys.path:
    sys.path.insert(0, str(_STANDALONE_ROOT))
_GENERATED_ROOT = _STANDALONE_ROOT.parent / "data" / "generated"


@lru_cache(maxsize=None)
def _default_alias_root() -> Path:
    """Ephemeral per-process alias root: avoids leaving a persistent
    _pkg_alias/ directory in the tracked standalone/ tree; removed
    automatically when the process exits. Set BIOTOX_PACKAGE_ALIAS_ROOT
    to override with a fixed, non-cleaned-up location instead."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="biotox_pkg_alias_"))
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)
    return tmp_dir


def _register_bundled_package(name: str, *locations: Path) -> None:
    """Make bundled standalone ``run/`` directories importable under
    original package names -- both in this process and in any subprocess
    or multiprocessing worker it spawns.

    An in-memory ``sys.modules`` entry alone is invisible to spawned
    worker processes (e.g. joblib/loky ridge-fitting workers): each
    starts a fresh interpreter that must resolve ``publication.X`` /
    ``stage1.X`` imports itself when unpickling a task, using only the
    standard import system. A regular package directory (one with an
    ``__init__.py``, as several bundled ``run/`` directories have) does
    not merge with same-named directories elsewhere on ``sys.path`` --
    only PEP 420 *namespace* packages do -- so pointing multiple
    ``sys.path`` entries at existing package directories would let
    only the first-found one's submodules resolve. Instead, this
    creates ONE namespace-package directory
    (``_STANDALONE_ROOT/_pkg_alias/<name>``, no ``__init__.py`` of its
    own) containing a flat, relative on-disk symlink to every ``.py``
    file from every ``location`` (so a ``publication`` alias can
    combine modules that physically live under both ``stage2`` and
    ``stage1``), and adds its PARENT directory to ``sys.path`` and to
    the ``PYTHONPATH`` environment variable. ``sys.path`` (not just
    ``PYTHONPATH``) must be mutated directly because multiprocessing's
    "spawn" start method (used by joblib/loky workers) propagates the
    parent's ``sys.path`` list to each worker rather than re-deriving
    it from ``PYTHONPATH``; ``PYTHONPATH`` is still set for plain
    ``subprocess.run`` stage launches, which DO read it at their own
    interpreter startup.
    """
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(location) for location in locations]
        sys.modules[name] = module

    alias_root = (
        Path(os.environ["BIOTOX_PACKAGE_ALIAS_ROOT"])
        if os.environ.get("BIOTOX_PACKAGE_ALIAS_ROOT")
        else _default_alias_root()
    )
    alias_dir = alias_root / name
    alias_dir.mkdir(parents=True, exist_ok=True)
    for location in locations:
        for source in sorted(location.glob("*.py")):
            if source.name == "__init__.py":
                continue
            link = alias_dir / source.name
            if not link.exists():
                link.symlink_to(os.path.relpath(source, alias_dir))
    if str(alias_root) not in sys.path:
        sys.path.insert(0, str(alias_root))
    existing_entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    merged = os.pathsep.join(
        entry for entry in [str(alias_root)] + existing_entries if entry
    )
    os.environ["PYTHONPATH"] = merged


from stage2.offset_ridge_fixed import (
    fixed_offset_ridge_objective_gradient,
    fit_fixed_offset_ridge_path,
    predict_fixed_offset_ridge,
)
from stage2.offset_ridge_lbfgs import fit_offset_intercept_lbfgs
from stage2.utils import (
    CELL_IDS,
    TOX21_TASKS,
    build_task_cell_frame,
    compute_binary_metrics,
    fit_offset_intercept_glm,
    fit_offset_ridge_glm,
    load_matched_split,
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
STAGE2_MODEL_SPEC = "frozen_offset_fixed_calibration_intercept_ridge_biology_v3"

COHORT_DIR = _GENERATED_ROOT / "molecular_prediction_and_ensemble" / "cohort"
GENE_FEATURES_DIR = _GENERATED_ROOT / "transcriptomic_residual_learning" / "gene_features_6h"
CHEM_OFFSET_PATH = _GENERATED_ROOT / "molecular_prediction_and_ensemble" / "deployed_k4" / "chem_offset_final.csv"
DEFAULT_OUT_DIR = _GENERATED_ROOT / "transcriptomic_residual_learning" / "primary_6h_20x5"
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
        return 0.0

    @property
    def beta(self) -> np.ndarray:
        return np.asarray(self.params, dtype=float)


@dataclass(frozen=True)
class Stage2FeatureScaling:
    """Fold-local Stage-2 gene scaler fit only on the supplied training matrix."""

    variant: str
    fit_matrix: str
    n_fit_rows: int
    n_apply_rows: int
    n_genes: int
    n_zero_sd_genes: int
    scale_reference_variant: str | None = None

    @property
    def order(self) -> str:
        if self.variant == "residualized" and self.scale_reference_variant is not None:
            return (
                "ceviche_residualize_then_scale_by_"
                f"{self.scale_reference_variant}_training_matrix_mean_sd"
            )
        if self.variant == "residualized":
            return "ceviche_residualize_then_fit_gene_mean_sd_on_training_matrix"
        return "raw_then_fit_gene_mean_sd_on_training_matrix"


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
            include_intercept=False,
            var_weights=sample_weight,
            maxiter=maxiter,
            cnvrg_tol=acceptance_gradient,
        )
    warned = any("ridge optimization may have failed" in str(item.message) for item in caught)
    params = np.asarray(result.params, dtype=float)
    objective, gradient = fixed_offset_ridge_objective_gradient(
        params,
        X,
        y,
        offset,
        alpha,
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
    if sample_weight is not None:
        raise ValueError(
            "The publication Stage-2 method uses unweighted BCE; sample weights "
            "are not supported by the fixed-calibration ridge path."
        )
    if solver == "lbfgs":
        return fit_fixed_offset_ridge_path(
            X,
            y,
            offset,
            alphas,
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
        return predict_fixed_offset_ridge(fitted, X, offset)[0]
    logits = _predict_logits(fitted, X, offset, solver)
    return expit(logits)


def _predict_logits(fitted, X, offset, solver: str) -> np.ndarray:
    if solver == "lbfgs":
        return predict_fixed_offset_ridge(fitted, X, offset)[1]
    X = np.asarray(X, dtype=np.float64)
    offset = np.asarray(offset, dtype=np.float64).reshape(-1)
    beta = np.asarray(fitted.beta, dtype=np.float64)
    logits = offset + X @ beta
    if not np.all(np.isfinite(logits)):
        raise RuntimeError("prediction produced non-finite logits")
    return logits


def _mean_bce_from_logits(y: np.ndarray, logits: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if y.shape[0] != logits.shape[0]:
        raise ValueError("y and logits must have the same length")
    loss = np.logaddexp(0.0, logits) - y * logits
    return float(np.mean(loss))


def _fit_apply_stage2_feature_scaler(
    variant: str,
    X_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    *,
    fit_matrix: str,
    scale_reference_train_raw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, Stage2FeatureScaling]:
    if variant not in VARIANT_SUFFIX:
        raise ValueError(f"unknown Stage-2 feature variant: {variant}")
    X_train_raw = np.asarray(X_train_raw, dtype=np.float64)
    X_test_raw = np.asarray(X_test_raw, dtype=np.float64)
    if X_train_raw.ndim != 2 or X_test_raw.ndim != 2:
        raise ValueError("Stage-2 feature matrices must be two-dimensional")
    if X_train_raw.shape[1] != X_test_raw.shape[1]:
        raise ValueError(
            "Stage-2 scaler cannot be applied across different gene dimensions"
        )
    # Fit moments in the coordinates of the representation being modeled.
    # The CEViChE-residualized branch therefore uses the residualized
    # training matrix itself. Borrowing raw/unadjusted moments would make the
    # saved beta vector incompatible with reconstruction from residualized
    # features and would confound ridge geometry with viability adjustment.
    # The optional reference argument remains for backward API compatibility,
    # but is intentionally ignored by the corrected implementation.
    fit_source = X_train_raw
    mu = fit_source.mean(axis=0)
    sd_raw = fit_source.std(axis=0)
    zero_sd = sd_raw < 1e-8
    sd = np.where(zero_sd, 1.0, sd_raw)
    X_train = (X_train_raw - mu) / sd
    X_test = (X_test_raw - mu) / sd
    if not np.all(np.isfinite(X_train)) or not np.all(np.isfinite(X_test)):
        raise ValueError("Stage-2 scaled feature matrices contain non-finite values")
    scaling = Stage2FeatureScaling(
        variant=variant,
        fit_matrix=fit_matrix,
        n_fit_rows=int(X_train_raw.shape[0]),
        n_apply_rows=int(X_test_raw.shape[0]),
        n_genes=int(X_train_raw.shape[1]),
        n_zero_sd_genes=int(np.sum(zero_sd)),
        scale_reference_variant=None,
    )
    return X_train, X_test, scaling


def _select_alpha_from_losses(
    losses_by_alpha: dict[float, list[float]],
) -> dict[str, Any] | None:
    if not losses_by_alpha:
        return None
    finite_losses = {}
    for alpha, losses in losses_by_alpha.items():
        if not np.isfinite(alpha) or not losses:
            continue
        alpha_losses = [float(loss) for loss in losses]
        if not all(np.isfinite(alpha_losses)):
            continue
        finite_losses[float(alpha)] = alpha_losses
    mean_loss = {
        alpha: float(np.mean(losses))
        for alpha, losses in finite_losses.items()
        if losses
    }
    if not mean_loss:
        return None

    alpha_min = min(mean_loss, key=mean_loss.get)
    losses_at_min = finite_losses[alpha_min]
    se_min = (
        float(np.std(losses_at_min, ddof=1) / np.sqrt(len(losses_at_min)))
        if len(losses_at_min) > 1
        else 0.0
    )
    threshold = mean_loss[alpha_min] + se_min
    ascending = sorted(mean_loss)
    eligible = [alpha for alpha in ascending if mean_loss[alpha] <= threshold]
    if not eligible:
        return None
    selected = max(eligible)

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
            if job.use_sample_weight:
                empty.error = (
                    "use_sample_weight=True is incompatible with the publication "
                    "Methods path, which uses unweighted BCE for calibration, "
                    "ridge fitting, and validation."
                )
                return empty
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
            X_train, X_test, outer_scaling = _fit_apply_stage2_feature_scaler(
                variant,
                X_train_raw,
                X_test_raw,
                fit_matrix="residualized_outer_training"
                if variant == "residualized"
                else "raw_outer_training",
                scale_reference_train_raw=scale_ref_train,
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
                X_inner_train, X_inner_valid, inner_scaling = _fit_apply_stage2_feature_scaler(
                    variant,
                    X_inner_train_raw,
                    X_inner_valid_raw,
                    fit_matrix="residualized_inner_training"
                    if variant == "residualized"
                    else "raw_inner_training",
                    scale_reference_train_raw=scale_ref_inner_train,
                )
                inner_calibration = fit_offset_intercept_lbfgs(
                    y_inner_train,
                    offset_train[inner_train_idx],
                )
                fixed_inner_train_offset = (
                    offset_train[inner_train_idx] + inner_calibration.intercept
                )
                fixed_inner_valid_offset = (
                    offset_train[inner_valid_idx] + inner_calibration.intercept
                )
                fitted_path = _fit_path(
                    job.solver,
                    X_inner_train,
                    y_inner_train,
                    fixed_inner_train_offset,
                    job.alpha_grid,
                    None,
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
                        validation_logit = _predict_logits(
                            fit,
                            X_inner_valid,
                            fixed_inner_valid_offset,
                            job.solver,
                        )
                        loss = _mean_bce_from_logits(
                            y_inner_valid,
                            validation_logit,
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
                            "stage2_scaling_order": inner_scaling.order,
                            "stage2_scaler_fit_matrix": inner_scaling.fit_matrix,
                            "stage2_scaler_fit_rows": inner_scaling.n_fit_rows,
                            "stage2_scaler_apply_rows": inner_scaling.n_apply_rows,
                            "stage2_scaler_n_genes": inner_scaling.n_genes,
                            "stage2_scaler_zero_sd_genes": inner_scaling.n_zero_sd_genes,
                            "calibration_intercept_fixed_for_beta": True,
                            "calibration_intercept_frozen": True,
                            "inner_calibration_intercept": inner_calibration.intercept,
                            "success": fit.success,
                            "validation_bce": loss,
                            "validation_deviance": 2.0 * loss,
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
            if job.solver == "lbfgs":
                null_fit = fit_offset_intercept_lbfgs(
                    y_train, offset_train
                )
                null_intercept = null_fit.intercept
                final_fit = _fit_path(
                    job.solver,
                    X_train,
                    y_train,
                    offset_train + null_intercept,
                    [selected_alpha],
                    None,
                    job.maxiter,
                    job.retry_maxiter,
                    job.gtol,
                    job.acceptance_gradient,
                    job.ftol,
                )[0]
            else:
                null_result = fit_offset_intercept_glm(
                    y_train, offset_train
                )
                null_intercept = float(np.asarray(null_result.params).reshape(-1)[0])
                final_fit = _statsmodels_fit(
                    X_train,
                    y_train,
                    offset_train + null_intercept,
                    selected_alpha,
                    None,
                    job.maxiter,
                    job.acceptance_gradient,
                )
            if not final_fit.success:
                empty.error = f"final selected-alpha fit failed: {final_fit.message}"
                empty.alpha_rows = alpha_rows
                return empty

            logit_stage1_test = offset_test
            logit_chem_test = offset_test + null_intercept
            logit_bio_test = _predict_logits(
                final_fit, X_test, logit_chem_test, job.solver
            )
            p_stage1_test = expit(offset_test)
            p_chem_test = expit(logit_chem_test)
            p_bio_test = expit(logit_bio_test)
            m_stage1 = compute_binary_metrics(y_test, p_stage1_test)
            m_chem = compute_binary_metrics(y_test, p_chem_test)
            m_bio = compute_binary_metrics(y_test, p_bio_test)
            log_loss_stage1_raw = _mean_bce_from_logits(y_test, logit_stage1_test)
            log_loss_chem = _mean_bce_from_logits(y_test, logit_chem_test)
            log_loss_bio = _mean_bce_from_logits(y_test, logit_bio_test)

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
                "stage2_scaling_order": outer_scaling.order,
                "stage2_scaler_fit_matrix": outer_scaling.fit_matrix,
                "stage2_scaler_fit_rows": outer_scaling.n_fit_rows,
                "stage2_scaler_apply_rows": outer_scaling.n_apply_rows,
                "stage2_scaler_n_genes": outer_scaling.n_genes,
                "stage2_scaler_zero_sd_genes": outer_scaling.n_zero_sd_genes,
                "intercept_penalized": False,
                "calibration_intercept_fixed_for_beta": True,
                "calibration_intercept_frozen": True,
                "use_sample_weight": job.use_sample_weight,
                "stage2_likelihood": "unweighted_publication_primary",
                "null_intercept": null_intercept,
                "bio_intercept": null_intercept,
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
                "log_loss_stage1_raw": log_loss_stage1_raw,
                "log_loss_chem": log_loss_chem,
                "log_loss_bio": log_loss_bio,
                "delta_log_loss": log_loss_bio - log_loss_chem,
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
                    "stage2_scaling_order": outer_scaling.order,
                    "stage2_scaler_fit_matrix": outer_scaling.fit_matrix,
                    "stage2_scaler_fit_rows": outer_scaling.n_fit_rows,
                    "stage2_scaler_apply_rows": outer_scaling.n_apply_rows,
                    "stage2_scaler_n_genes": outer_scaling.n_genes,
                    "stage2_scaler_zero_sd_genes": outer_scaling.n_zero_sd_genes,
                    "calibration_intercept_fixed_for_beta": True,
                    "calibration_intercept_frozen": True,
                    "solver": job.solver,
                    "y": float(prepared.y[index]),
                    "p_chem_stage1_raw": float(p_stage1_test[position]),
                    "p_chem_only": float(p_chem_test[position]),
                    "p_chem_bio": float(p_bio_test[position]),
                    "logit_stage1_raw": float(logit_stage1_test[position]),
                    "logit_chem_only": float(logit_chem_test[position]),
                    "logit_chem_bio": float(logit_bio_test[position]),
                    "transcriptomic_logit": float(
                        logit_bio_test[position] - logit_chem_test[position]
                    ),
                    "null_intercept": null_intercept,
                    "bio_intercept": null_intercept,
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
    if use_sample_weight:
        raise ValueError(
            "use_sample_weight=True is incompatible with the publication Methods "
            "implementation. Stage 2 uses unweighted BCE for calibration, beta-only "
            "ridge fitting, and inner validation."
        )
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
        "pipeline": "calibrated_stage2_fixed_calibration_v3",
        "stage2_model_spec": STAGE2_MODEL_SPEC,
        "solver": solver,
        "chemical_offset_path": str(chem_offset_path.resolve()),
        "chemical_offset_frozen": True,
        "chemical_offset_coefficient": 1.0,
        "intercept_penalized": False,
        "calibration_intercept_policy": (
            "fit one unpenalized offset-only intercept on each training subset, "
            "then freeze it before fitting transcriptomic ridge coefficients"
        ),
        "calibration_intercept_frozen": True,
        "ridge_fit_policy": (
            "fit beta only with the full chemical-plus-calibration logit as a "
            "fixed offset; no intercept is optimized in the ridge objective"
        ),
        "alpha_selection_policy": (
            "select the largest alpha whose mean inner-validation BCE is within "
            "one standard error of the minimum mean BCE"
        ),
        "validation_loss": "unweighted_mean_binary_cross_entropy_from_logits",
        "stage2_likelihood": "unweighted_publication_primary",
        "sample_weight_policy": "forbidden_for_publication_methods_path",
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
        "stage2_scaling_policy": (
            "raw branch: fit gene-wise mean/SD on each raw training matrix; "
            "residualized branch: fixed external CEViChE projection is already "
            "removed before fold-local gene-wise mean/SD fitting; validation and "
            "outer-test matrices only receive the fitted training scaler"
        ),
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
        f"[fixed-calibration-v3] {len(jobs)}/{expected_outer_fold_jobs} "
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
    # CEViChE-residualized variant is deprecated; unadjusted ("standardized") is
    # now the only supported mode. Original choice kept commented for provenance.
    # parser.add_argument("--variant", choices=["standardized", "residualized", "both"], default="both")
    parser.add_argument("--variant", choices=["standardized"], default="standardized")
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
