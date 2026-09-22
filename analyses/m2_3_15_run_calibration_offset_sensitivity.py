"""Run the clean C0/D0/C1/D1 Stage-2 calibration-ridge ablation.

The source 20x5 run supplies immutable outer-test membership and frozen
chemical predictions. This script refits only the two fixed-offset biological
ridge branches:

    C0: eta_chem
    D0: eta_chem + a0
    C1: eta_chem + X beta_C
    D1: eta_chem + a0 + X beta_D

For D1, a0 is fitted on the relevant training partition and then held fixed;
no second intercept is estimated. C1 and D1 select ridge penalties
independently within each outer-training partition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
import scipy.special as scipy_special
import sklearn.metrics as sk_metrics
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from publication import nested_cv_offset_logistic_calibrated_v2 as calibrated  # noqa: E402
from publication.offset_ridge_fixed_v1 import (  # noqa: E402
    FixedOffsetRidgeFit,
    fit_fixed_offset_ridge_lbfgs,
    fit_fixed_offset_ridge_path,
    predict_fixed_offset_ridge,
)
from publication.offset_ridge_lbfgs import fit_offset_intercept_lbfgs  # noqa: E402
from publication.repeated_cv_inference import (  # noqa: E402
    repeated_cv_nadeau_bengio_test,
)
from publication.utils import (  # noqa: E402
    _weighted_binomial_deviance,
    standardize_train_test,
    stratified_scaffold_group_kfold,
)


SOURCE_RUN_DIR = (
    HERE
    / "_run_output"
    / "nested_cv_k4_calibrated_lbfgs_repeated20x5_combined_v1"
)
DEFAULT_OUT_DIR = (
    HERE
    / "_run_output"
    / "stage2_calibration_ridge_ablation_20x5_v1"
)
SOURCE_GENE_FEATURES_DIR = (
    HERE / "_run_output" / "gene_features_consistent_v1"
)
MODEL_SPEC = "fixed_chemical_offset_fixed_calibration_intercept_ridge_biology_v1"
CONTRASTS = (
    ("c1_minus_c0", "c1", "c0"),
    ("d1_minus_d0", "d1", "d0"),
    ("d1_minus_c1", "d1", "c1"),
)
METRICS = ("auprc", "auroc", "log_loss", "brier")


@dataclass(frozen=True)
class FoldAssignment:
    repeat_id: int
    fold_id: int
    outer_seed_requested: int
    outer_split_seed: int
    outer_split_signature: str
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class FoldJob:
    prepared: calibrated.PreparedTaskCell
    assignment: FoldAssignment
    alpha_grid: tuple[float, ...]
    maxiter: int
    retry_maxiter: int
    gtol: float
    acceptance_gradient: float
    ftol: float


@dataclass
class FoldOutcome:
    success: bool
    error: str
    fold_row: dict[str, Any] | None
    oof_rows: list[dict[str, Any]]
    alpha_rows: list[dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_csv_list(value: str | None, allowed: list[str]) -> list[str]:
    if value is None:
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected).difference(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return selected


def _source_oof_path(
    source_run_dir: Path,
    variant: str,
    task: str,
    cell: str,
) -> Path:
    return source_run_dir / (
        f"repeated_nested_cv_oof_predictions_{variant}_{task}__{cell}.csv"
    )


def _load_fold_assignments(
    source_run_dir: Path,
    prepared: calibrated.PreparedTaskCell,
    *,
    expected_repeats: int,
    repeat_limit: int | None = None,
) -> list[FoldAssignment]:
    path = _source_oof_path(
        source_run_dir, prepared.variant, prepared.task, prepared.cell
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    source = pd.read_csv(path)
    required = {
        "repeat_id",
        "fold_id",
        "outer_seed_requested",
        "outer_split_seed",
        "outer_split_signature",
        "ik",
        "y",
        "p_chem_stage1_raw",
        "p_chem_only",
        "null_intercept",
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    key = ["repeat_id", "fold_id", "ik"]
    if source.duplicated(key).any():
        raise ValueError(f"Duplicate source OOF rows: {path}")
    if source["repeat_id"].nunique() != expected_repeats:
        raise ValueError(
            f"{path} has {source['repeat_id'].nunique()} repeats; "
            f"expected {expected_repeats}"
        )

    index_by_ik = {ik: index for index, ik in enumerate(prepared.ik)}
    all_indices = np.arange(len(prepared.ik), dtype=int)
    assignments: list[FoldAssignment] = []
    for (repeat_id, fold_id), group in source.groupby(
        ["repeat_id", "fold_id"], sort=True
    ):
        unknown = sorted(set(group["ik"]).difference(index_by_ik))
        if unknown:
            raise ValueError(f"Source OOF contains unknown compounds: {unknown[:3]}")
        test_idx = np.asarray(
            [index_by_ik[ik] for ik in group["ik"]], dtype=int
        )
        train_idx = np.setdiff1d(all_indices, test_idx, assume_unique=False)
        expected_y = prepared.y[test_idx]
        observed_y = group["y"].to_numpy(dtype=float)
        expected_raw = scipy_special.expit(prepared.offset[test_idx])
        observed_raw = group["p_chem_stage1_raw"].to_numpy(dtype=float)
        if not np.array_equal(expected_y, observed_y):
            raise ValueError(f"Label mismatch in source fold {repeat_id}/{fold_id}")
        if not np.allclose(expected_raw, observed_raw, atol=1e-12, rtol=1e-10):
            raise ValueError(
                f"Frozen chemical prediction mismatch in fold {repeat_id}/{fold_id}"
            )
        intercepts = group["null_intercept"].to_numpy(dtype=float)
        if np.unique(intercepts).size != 1:
            raise ValueError(f"Inconsistent source intercept in {repeat_id}/{fold_id}")
        expected_d0 = scipy_special.expit(
            prepared.offset[test_idx] + intercepts[0]
        )
        observed_d0 = group["p_chem_only"].to_numpy(dtype=float)
        if not np.allclose(expected_d0, observed_d0, atol=1e-12, rtol=1e-10):
            raise ValueError(f"Source D0 prediction mismatch in {repeat_id}/{fold_id}")
        assignments.append(
            FoldAssignment(
                repeat_id=int(repeat_id),
                fold_id=int(fold_id),
                outer_seed_requested=int(group["outer_seed_requested"].iloc[0]),
                outer_split_seed=int(group["outer_split_seed"].iloc[0]),
                outer_split_signature=str(
                    group["outer_split_signature"].iloc[0]
                ),
                train_idx=train_idx,
                test_idx=test_idx,
            )
        )
    if repeat_limit is not None:
        if repeat_limit < 1 or repeat_limit > expected_repeats:
            raise ValueError("repeat_limit must be between 1 and expected_repeats")
        assignments = [
            assignment
            for assignment in assignments
            if assignment.repeat_id < repeat_limit
        ]
    analysis_repeats = expected_repeats if repeat_limit is None else repeat_limit
    expected_folds = analysis_repeats * calibrated.OUTER_K
    if len(assignments) != expected_folds:
        raise ValueError(f"Found {len(assignments)} folds; expected {expected_folds}")
    return assignments


def _binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return {
        "auprc": float(sk_metrics.average_precision_score(y, probability)),
        "auroc": float(sk_metrics.roc_auc_score(y, probability)),
        "log_loss": float(
            sk_metrics.log_loss(y, probability, labels=[0.0, 1.0])
        ),
        "brier": float(sk_metrics.brier_score_loss(y, probability)),
    }


def _fit_branch_path(
    X: np.ndarray,
    y: np.ndarray,
    offset: np.ndarray,
    job: FoldJob,
) -> list[FixedOffsetRidgeFit]:
    return fit_fixed_offset_ridge_path(
        X,
        y,
        offset,
        job.alpha_grid,
        maxiter=job.maxiter,
        retry_maxiter=job.retry_maxiter,
        gtol=job.gtol,
        acceptance_gradient=job.acceptance_gradient,
        ftol=job.ftol,
    )


def _run_fold(job: FoldJob) -> FoldOutcome:
    prepared = job.prepared
    assignment = job.assignment
    try:
        with threadpool_limits(limits=1):
            train_idx = assignment.train_idx
            test_idx = assignment.test_idx
            y_train = prepared.y[train_idx]
            y_test = prepared.y[test_idx]
            X_train_raw = prepared.X_raw[train_idx]
            X_test_raw = prepared.X_raw[test_idx]
            if prepared.variant == "standardized":
                X_train, X_test = standardize_train_test(
                    X_train_raw, X_test_raw
                )
            else:
                X_train, X_test = X_train_raw, X_test_raw
            offset_train = prepared.offset[train_idx]
            offset_test = prepared.offset[test_idx]
            smiles_train = [prepared.smiles[index] for index in train_idx]

            inner_folds = stratified_scaffold_group_kfold(
                smiles_train,
                y_train,
                k=calibrated.INNER_CV_K,
                seed=calibrated.INNER_CV_SEED,
            )
            if len(inner_folds) != calibrated.INNER_CV_K:
                raise ValueError("Inner splitter returned an incomplete partition")

            losses = {
                "c1": {alpha: [] for alpha in job.alpha_grid},
                "d1": {alpha: [] for alpha in job.alpha_grid},
            }
            alpha_rows: list[dict[str, Any]] = []
            for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
                inner_folds
            ):
                y_inner_train = y_train[inner_train_idx]
                y_inner_valid = y_train[inner_valid_idx]
                if (
                    np.unique(y_inner_train).size != 2
                    or np.unique(y_inner_valid).size != 2
                ):
                    raise ValueError(f"Inner fold {inner_fold} is not evaluable")
                X_inner_train_raw = X_train_raw[inner_train_idx]
                X_inner_valid_raw = X_train_raw[inner_valid_idx]
                if prepared.variant == "standardized":
                    X_inner_train, X_inner_valid = standardize_train_test(
                        X_inner_train_raw, X_inner_valid_raw
                    )
                else:
                    X_inner_train, X_inner_valid = (
                        X_inner_train_raw,
                        X_inner_valid_raw,
                    )
                raw_inner_train = offset_train[inner_train_idx]
                raw_inner_valid = offset_train[inner_valid_idx]
                d0_inner = fit_offset_intercept_lbfgs(
                    y_inner_train, raw_inner_train
                ).intercept
                branch_offsets = {
                    "c1": (raw_inner_train, raw_inner_valid),
                    "d1": (
                        raw_inner_train + d0_inner,
                        raw_inner_valid + d0_inner,
                    ),
                }
                for branch, (fit_offset, valid_offset) in branch_offsets.items():
                    path = _fit_branch_path(
                        X_inner_train, y_inner_train, fit_offset, job
                    )
                    by_alpha = {fit.alpha: fit for fit in path}
                    for alpha in job.alpha_grid:
                        fit = by_alpha[alpha]
                        validation_deviance = float("nan")
                        if fit.success:
                            probability = predict_fixed_offset_ridge(
                                fit, X_inner_valid, valid_offset
                            )[0]
                            validation_deviance = _weighted_binomial_deviance(
                                y_inner_valid, probability, None
                            )
                            losses[branch][alpha].append(validation_deviance)
                        alpha_rows.append(
                            {
                                "task": prepared.task,
                                "cell": prepared.cell,
                                "variant": prepared.variant,
                                "repeat_id": assignment.repeat_id,
                                "fold_id": assignment.fold_id,
                                "inner_fold": inner_fold,
                                "branch": branch,
                                "alpha": alpha,
                                "success": fit.success,
                                "validation_deviance": validation_deviance,
                                "d0_inner_intercept": d0_inner,
                                "warm_started": fit.warm_started,
                                "retry_used": fit.retry_used,
                                "n_iterations": fit.n_iterations,
                                "gradient_inf_norm": fit.gradient_inf_norm,
                                "objective": fit.objective,
                                "elapsed_seconds": fit.elapsed_seconds,
                                "message": fit.message,
                            }
                        )

            selection: dict[str, dict[str, Any]] = {}
            for branch in ("c1", "d1"):
                complete = {
                    alpha: values
                    for alpha, values in losses[branch].items()
                    if len(values) == calibrated.INNER_CV_K
                }
                info = calibrated._select_alpha_from_losses(complete)
                if info is None:
                    raise RuntimeError(f"No complete alpha path for {branch}")
                selection[branch] = info

            d0_fit = fit_offset_intercept_lbfgs(y_train, offset_train)
            d0_intercept = d0_fit.intercept
            c1_fit = fit_fixed_offset_ridge_lbfgs(
                X_train,
                y_train,
                offset_train,
                selection["c1"]["alpha_star"],
                maxiter=job.maxiter,
                retry_maxiter=job.retry_maxiter,
                gtol=job.gtol,
                acceptance_gradient=job.acceptance_gradient,
                ftol=job.ftol,
            )
            d1_fit = fit_fixed_offset_ridge_lbfgs(
                X_train,
                y_train,
                offset_train + d0_intercept,
                selection["d1"]["alpha_star"],
                maxiter=job.maxiter,
                retry_maxiter=job.retry_maxiter,
                gtol=job.gtol,
                acceptance_gradient=job.acceptance_gradient,
                ftol=job.ftol,
            )
            if not c1_fit.success or not d1_fit.success:
                raise RuntimeError(
                    f"Final fit failed: C1={c1_fit.message}; D1={d1_fit.message}"
                )

            probabilities = {
                "c0": scipy_special.expit(offset_test),
                "d0": scipy_special.expit(offset_test + d0_intercept),
                "c1": predict_fixed_offset_ridge(
                    c1_fit, X_test, offset_test
                )[0],
                "d1": predict_fixed_offset_ridge(
                    d1_fit, X_test, offset_test + d0_intercept
                )[0],
            }
            metrics = {
                model: _binary_metrics(y_test, probability)
                for model, probability in probabilities.items()
            }
            if not np.isclose(metrics["c0"]["auprc"], metrics["d0"]["auprc"]):
                raise AssertionError("C0 and D0 AUPRC differ despite constant shift")
            if not np.isclose(metrics["c0"]["auroc"], metrics["d0"]["auroc"]):
                raise AssertionError("C0 and D0 AUROC differ despite constant shift")

            fold_row: dict[str, Any] = {
                "task": prepared.task,
                "cell": prepared.cell,
                "variant": prepared.variant,
                "repeat_id": assignment.repeat_id,
                "fold_id": assignment.fold_id,
                "outer_seed_requested": assignment.outer_seed_requested,
                "outer_split_seed": assignment.outer_split_seed,
                "outer_split_signature": assignment.outer_split_signature,
                "n_outer_train": len(train_idx),
                "n_outer_test": len(test_idx),
                "chemical_offset_frozen": True,
                "chemical_offset_coefficient": 1.0,
                "d0_intercept": d0_intercept,
                "d1_intercept_refit": False,
                "c1_has_intercept": False,
                "stage2_likelihood": "unweighted",
                "stage2_model_spec": MODEL_SPEC,
            }
            for branch, fit in (("c1", c1_fit), ("d1", d1_fit)):
                info = selection[branch]
                fold_row.update(
                    {
                        f"alpha_{branch}": info["alpha_star"],
                        f"alpha_min_loss_{branch}": info["alpha_min_loss"],
                        f"alpha_boundary_hit_{branch}": info[
                            "alpha_boundary_hit"
                        ],
                        f"beta_saturated_no_signal_{branch}": info[
                            "beta_saturated_no_bio_signal"
                        ],
                        f"beta_l2_norm_{branch}": float(
                            np.linalg.norm(fit.beta)
                        ),
                        f"final_gradient_inf_norm_{branch}": (
                            fit.gradient_inf_norm
                        ),
                        f"final_retry_used_{branch}": fit.retry_used,
                    }
                )
            for model in ("c0", "d0", "c1", "d1"):
                for metric in METRICS:
                    fold_row[f"{metric}_{model}"] = metrics[model][metric]
            for contrast, upper, lower in CONTRASTS:
                for metric in METRICS:
                    fold_row[f"delta_{metric}_{contrast}"] = (
                        metrics[upper][metric] - metrics[lower][metric]
                    )

            oof_rows = []
            for position, index in enumerate(test_idx):
                row = {
                    "task": prepared.task,
                    "cell": prepared.cell,
                    "variant": prepared.variant,
                    "repeat_id": assignment.repeat_id,
                    "fold_id": assignment.fold_id,
                    "outer_split_signature": assignment.outer_split_signature,
                    "ik": prepared.ik[index],
                    "y": float(prepared.y[index]),
                    "d0_intercept": d0_intercept,
                }
                for model, probability in probabilities.items():
                    row[f"p_{model}"] = float(probability[position])
                oof_rows.append(row)
            return FoldOutcome(True, "", fold_row, oof_rows, alpha_rows)
    except Exception as exc:
        return FoldOutcome(False, f"{type(exc).__name__}: {exc}", None, [], [])


def _bh_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(dtype=float)
    result = np.full(p.shape, np.nan, dtype=float)
    finite = np.isfinite(p)
    observed = p[finite]
    if observed.size == 0:
        return result
    order = np.argsort(observed)
    ranked = observed[order]
    adjusted = ranked * observed.size / np.arange(1, observed.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.clip(adjusted, 0.0, 1.0)
    result[finite] = restored
    return result


def _summarize(folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["variant", "task", "cell"]
    for keys, group in folds.groupby(group_cols, sort=False):
        row: dict[str, Any] = dict(zip(group_cols, keys))
        row.update(
            {
                "n_repeats": int(group["repeat_id"].nunique()),
                "n_outer_folds": int(len(group)),
                "n_unique_compounds": int(
                    group["n_outer_train"].iloc[0]
                    + group["n_outer_test"].iloc[0]
                ),
            }
        )
        for model in ("c0", "d0", "c1", "d1"):
            for metric in METRICS:
                row[f"mean_{metric}_{model}"] = float(
                    group[f"{metric}_{model}"].mean()
                )
        for contrast, _, _ in CONTRASTS:
            for metric in METRICS:
                column = f"delta_{metric}_{contrast}"
                row[f"mean_{column}"] = float(group[column].mean())
                row[f"sd_{column}"] = float(group[column].std(ddof=1))
                nb = repeated_cv_nadeau_bengio_test(
                    group[column].to_numpy(dtype=float),
                    group["n_outer_train"].to_numpy(dtype=float),
                    group["n_outer_test"].to_numpy(dtype=float),
                )
                for key, value in nb.items():
                    row[f"nb_{metric}_{contrast}_{key}"] = value
        rows.append(row)
    summary = pd.DataFrame(rows)
    for variant in summary["variant"].unique():
        for cell in summary["cell"].unique():
            mask = summary["variant"].eq(variant) & summary["cell"].eq(cell)
            for contrast, _, _ in CONTRASTS:
                for metric in ("auprc", "auroc"):
                    p_col = f"nb_{metric}_{contrast}_p_value"
                    q_col = f"bh_q_{metric}_{contrast}"
                    summary.loc[mask, q_col] = _bh_adjust(
                        summary.loc[mask, p_col]
                    )
    return summary.sort_values(["variant", "task", "cell"]).reset_index(drop=True)


def _repeat_means(folds: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        column
        for column in folds.columns
        if column.startswith("auprc_")
        or column.startswith("auroc_")
        or column.startswith("log_loss_")
        or column.startswith("brier_")
        or column.startswith("delta_")
    ]
    return (
        folds.groupby(["variant", "task", "cell", "repeat_id"], as_index=False)[
            numeric
        ]
        .mean()
        .sort_values(["variant", "task", "cell", "repeat_id"])
    )


def run(args: argparse.Namespace) -> None:
    output_dir = args.out_dir.resolve()
    source_run_dir = args.source_run_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
    if not (source_run_dir / "RUN_COMPLETE.json").is_file():
        raise FileNotFoundError(
            f"Source run is not complete: {source_run_dir / 'RUN_COMPLETE.json'}"
        )
    tasks = _parse_csv_list(args.tasks, calibrated.TOX21_TASKS)
    cells = _parse_csv_list(args.cells, calibrated.CELL_IDS)
    variants = (
        ["standardized", "residualized"]
        if args.variant == "both"
        else [args.variant]
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    offsets = calibrated.validate_chem_offsets(
        calibrated.load_chem_offsets(args.chem_offset_path.resolve()), tasks
    )
    alpha_values = calibrated.alpha_grid(args.points_per_decade)
    resolved_jobs = min(
        max(1, effective_n_jobs(args.n_jobs)),
        (args.repeat_limit or args.expected_repeats)
        * calibrated.OUTER_K
        * len(variants),
    )
    analysis_repeats = args.repeat_limit or args.expected_repeats
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "stage2_calibration_ridge_ablation_v1",
        "model_spec": MODEL_SPEC,
        "source_run_dir": str(source_run_dir),
        "source_run_complete_sha256": _sha256(
            source_run_dir / "RUN_COMPLETE.json"
        ),
        "chemical_offset_path": str(args.chem_offset_path.resolve()),
        "chemical_offset_sha256": _sha256(args.chem_offset_path.resolve()),
        "chemical_offset_frozen": True,
        "chemical_offset_coefficient": 1.0,
        "d1_uses_fixed_d0_intercept": True,
        "d1_refits_intercept": False,
        "c1_has_intercept": False,
        "stage2_likelihood": "unweighted",
        "outer_membership": "copied_exactly_from_source_oof_predictions",
        "inner_folds": calibrated.INNER_CV_K,
        "inner_seed": calibrated.INNER_CV_SEED,
        "alpha_grid": list(alpha_values),
        "alpha_selected_independently_for_c1_and_d1": True,
        "variants": variants,
        "tasks": tasks,
        "cells": cells,
        "expected_repeats": args.expected_repeats,
        "analysis_repeats": analysis_repeats,
        "repeat_limit": args.repeat_limit,
        "requested_n_jobs": args.n_jobs,
        "resolved_n_jobs": resolved_jobs,
        "analysis_complete": False,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    all_fold_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    total_groups = len(tasks) * len(cells)
    group_index = 0
    for task in tasks:
        for cell in cells:
            group_index += 1
            group_outcomes: dict[str, list[FoldOutcome]] = {}
            for variant in variants:
                prepared = calibrated.prepare_task_cell(
                    task,
                    cell,
                    variant,
                    offsets,
                    outer_seed=calibrated.OUTER_SEED,
                    cohort_dir=args.cohort_dir.resolve(),
                    gene_features_dir=args.gene_features_dir.resolve(),
                )
                assignments = _load_fold_assignments(
                    source_run_dir,
                    prepared,
                    expected_repeats=args.expected_repeats,
                    repeat_limit=args.repeat_limit,
                )
                jobs = [
                    FoldJob(
                        prepared=prepared,
                        assignment=assignment,
                        alpha_grid=alpha_values,
                        maxiter=args.maxiter,
                        retry_maxiter=args.retry_maxiter,
                        gtol=args.gtol,
                        acceptance_gradient=args.acceptance_gradient,
                        ftol=args.ftol,
                    )
                    for assignment in assignments
                ]
                print(
                    f"[calibration-ridge] group {group_index}/{total_groups} "
                    f"{task}/{cell}/{variant}: {len(jobs)} folds"
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
                    )(delayed(_run_fold)(job) for job in jobs)
                group_outcomes[variant] = outcomes

            for variant, outcomes in group_outcomes.items():
                successful = [outcome for outcome in outcomes if outcome.success]
                for outcome in outcomes:
                    if not outcome.success:
                        failures.append(
                            {
                                "task": task,
                                "cell": cell,
                                "variant": variant,
                                "error": outcome.error,
                            }
                        )
                if len(successful) != analysis_repeats * calibrated.OUTER_K:
                    continue
                fold_df = pd.DataFrame(
                    [outcome.fold_row for outcome in successful]
                ).sort_values(["repeat_id", "fold_id"])
                oof_df = pd.DataFrame(
                    [row for outcome in successful for row in outcome.oof_rows]
                ).sort_values(["repeat_id", "fold_id", "ik"])
                alpha_df = pd.DataFrame(
                    [row for outcome in successful for row in outcome.alpha_rows]
                ).sort_values(
                    ["repeat_id", "fold_id", "branch", "inner_fold", "alpha"],
                    ascending=[True, True, True, True, False],
                )
                stem = f"{variant}_{task}__{cell}"
                fold_df.to_csv(
                    output_dir / f"calibration_ridge_folds_{stem}.csv",
                    index=False,
                )
                oof_df.to_csv(
                    output_dir / f"calibration_ridge_oof_{stem}.csv",
                    index=False,
                )
                alpha_df.to_csv(
                    output_dir / f"calibration_ridge_alpha_{stem}.csv",
                    index=False,
                )
                all_fold_frames.append(fold_df)

    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "outer_fold_failures.csv", index=False
        )
    if not all_fold_frames:
        raise RuntimeError("No complete calibration-ridge groups were produced")
    all_folds = pd.concat(all_fold_frames, ignore_index=True)
    all_folds.to_csv(output_dir / "calibration_ridge_folds_all.csv", index=False)
    summary = _summarize(all_folds)
    summary.to_csv(output_dir / "calibration_ridge_summary_all.csv", index=False)
    repeat_means = _repeat_means(all_folds)
    repeat_means.to_csv(
        output_dir / "calibration_ridge_repeat_means.csv", index=False
    )

    expected_groups = len(tasks) * len(cells) * len(variants)
    complete = bool(
        not failures
        and len(all_fold_frames) == expected_groups
        and len(all_folds)
        == expected_groups * analysis_repeats * calibrated.OUTER_K
    )
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_complete_groups": len(all_fold_frames),
            "n_expected_groups": expected_groups,
            "n_outer_fold_rows": len(all_folds),
            "n_failures": len(failures),
            "analysis_complete": complete,
        }
    )
    completion = "RUN_COMPLETE.json" if complete else "RUN_INCOMPLETE.json"
    (output_dir / completion).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if not complete:
        raise RuntimeError(
            f"Calibration-ridge ablation incomplete; see {output_dir}"
        )
    print(f"[calibration-ridge] complete: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run clean C0/D0/C1/D1 Stage-2 calibration-ridge ablation."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-run-dir", type=Path, default=SOURCE_RUN_DIR)
    parser.add_argument(
        "--chem-offset-path", type=Path, default=calibrated.CHEM_OFFSET_PATH
    )
    parser.add_argument("--cohort-dir", type=Path, default=calibrated.COHORT_DIR)
    parser.add_argument(
        "--gene-features-dir",
        type=Path,
        default=SOURCE_GENE_FEATURES_DIR,
    )
    parser.add_argument(
        "--variant",
        choices=["standardized", "residualized", "both"],
        default="both",
    )
    parser.add_argument("--tasks")
    parser.add_argument("--cells")
    parser.add_argument("--expected-repeats", type=int, default=20)
    parser.add_argument(
        "--repeat-limit",
        type=int,
        help="Use only the first N archived repeats for a versioned smoke run.",
    )
    parser.add_argument("--n-jobs", type=int, default=30)
    parser.add_argument("--points-per-decade", type=float, default=2.0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--retry-maxiter", type=int, default=2000)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--acceptance-gradient", type=float, default=1e-5)
    parser.add_argument("--ftol", type=float, default=1e-14)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
