"""Repeated calibrated Stage-2 nested scaffold CV.

This module reuses the audited calibrated-v2 outer-fold fitter without
changing its statistical model:

    logit(P(Y=1)) = eta_chem + intercept + X_bio @ beta

The Stage-1 chemical logit is a globally frozen offset with coefficient one,
the intercept is unpenalized, and only biological coefficients receive ridge
regularization. Repeats vary the outer scaffold partition, not model
initialization. Every repeat reruns the complete inner-CV alpha selection.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs, parallel_config

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from publication import nested_cv_offset_logistic_calibrated_v2 as calibrated  # noqa: E402
from publication.repeated_cv_inference import (  # noqa: E402
    repeated_cv_nadeau_bengio_test,
)
from publication.utils import stratified_scaffold_group_kfold  # noqa: E402


DEFAULT_OUTER_SEEDS = (
    2042,
    3042,
    4042,
    5042,
    6042,
    7042,
    8042,
    9042,
    10042,
    11042,
    12042,
    13042,
    14042,
    15042,
    16042,
    17042,
    18042,
    19042,
    20042,
    21042,
)
DEFAULT_SEED_SEARCH_ATTEMPTS = 1000
PIPELINE_NAME = "calibrated_stage2_repeated_nested_cv_lbfgs_v1"
@dataclass(frozen=True)
class RepeatSplit:
    repeat_id: int
    requested_seed: int
    actual_seed: int
    signature: str
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]


@dataclass(frozen=True)
class RepeatedPreparedSet:
    repeat: RepeatSplit
    prepared: calibrated.PreparedTaskCell


@dataclass(frozen=True)
class RepeatedJob:
    repeat: RepeatSplit
    job: calibrated.OuterFoldJob


@dataclass
class RepeatedOutcome:
    repeat: RepeatSplit
    outcome: calibrated.OuterFoldOutcome


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gene_feature_checksums(
    gene_features_dir: Path,
    cells: list[str],
    variants: list[str],
) -> dict[str, str]:
    suffix_by_variant = {
        "standardized": "raw",
        "residualized": "residualized",
    }
    checksums: dict[str, str] = {}
    for cell in cells:
        for variant in variants:
            suffix = suffix_by_variant[variant]
            path = gene_features_dir / f"{cell}_gene_features_{suffix}.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            checksums[path.name] = _file_sha256(path)
    feature_manifest = gene_features_dir / "FEATURE_MANIFEST.json"
    if feature_manifest.is_file():
        checksums[feature_manifest.name] = _file_sha256(feature_manifest)
    compatibility_audit = (
        gene_features_dir / "COMPATIBILITY_WITH_10X5_AUDIT.json"
    )
    if compatibility_audit.is_file():
        checksums[compatibility_audit.name] = _file_sha256(
            compatibility_audit
        )
    return checksums


def _validate_stage1_deployment_report(
    path: Path,
    tasks: list[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = pd.read_csv(path)
    required = {"task", "rule", "k", "ensemble_candidates"}
    missing = sorted(required.difference(report.columns))
    if missing:
        raise ValueError(f"Stage-1 deployment report is missing: {missing}")
    selected = report.loc[report["task"].isin(tasks)].copy()
    if set(selected["task"]) != set(tasks):
        raise ValueError("Stage-1 deployment report does not cover all tasks")
    if set(selected["rule"]) != {"mean"} or set(selected["k"]) != {4}:
        raise ValueError("Stage-1 deployment must be the fixed K=4 mean ensemble")
    if selected["ensemble_candidates"].isna().any():
        raise ValueError("Stage-1 deployment membership is incomplete")
    member_counts = selected["ensemble_candidates"].astype(str).map(
        lambda value: len([item for item in value.split(";") if item])
    )
    if not np.all(member_counts == 4):
        raise ValueError("Every Stage-1 task must record exactly four members")
    return selected


def parse_outer_seeds(value: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        seeds = tuple(int(item) for item in value)
    if len(seeds) < 2:
        raise ValueError("Repeated CV requires at least two outer split seeds")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Outer split seeds must be unique")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Outer split seeds must be non-negative")
    return seeds


def validate_seed_windows(seeds: tuple[int, ...], attempts: int) -> None:
    if attempts < 1:
        raise ValueError("seed_search_attempts must be positive")
    ordered = sorted(seeds)
    for left, right in zip(ordered, ordered[1:]):
        if right - left < attempts:
            raise ValueError(
                "Outer-seed search windows overlap. Space requested seeds by at "
                f"least {attempts}; conflicting seeds: {left}, {right}"
            )


def _fold_partition_signature(
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    compound_ids: tuple[str, ...],
) -> str:
    test_partitions = sorted(
        tuple(sorted(compound_ids[int(index)] for index in test_idx))
        for _, test_idx in folds
    )
    payload = json.dumps(test_partitions, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_partition(
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    y: np.ndarray,
    outer_k: int,
    min_class_count: int,
) -> None:
    if len(folds) != outer_k:
        raise ValueError(f"Expected {outer_k} outer folds, found {len(folds)}")
    n = len(y)
    test_counts = np.zeros(n, dtype=int)
    for train_idx, test_idx in folds:
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError("Outer partition contains an empty train or test fold")
        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError("Outer train and test indices overlap")
        if len(train_idx) + len(test_idx) != n:
            raise ValueError("Outer train and test indices do not cover the cohort")
        test_counts[test_idx] += 1
        for indices in (train_idx, test_idx):
            for class_value in (0.0, 1.0):
                if np.sum(y[indices] == class_value) < min_class_count:
                    raise ValueError(
                        "Outer partition does not satisfy the predeclared "
                        "minimum class count"
                    )
    if not np.all(test_counts == 1):
        raise ValueError(
            "Each compound must occur in exactly one outer-test fold per repeat"
        )


def find_unique_evaluable_repeat(
    prepared: calibrated.PreparedTaskCell,
    repeat_id: int,
    requested_seed: int,
    used_signatures: set[str],
    *,
    outer_k: int = calibrated.OUTER_K,
    max_attempts: int = DEFAULT_SEED_SEARCH_ATTEMPTS,
    min_class_count: int = calibrated.MIN_CLASS_PER_OUTER_PARTITION,
) -> RepeatSplit:
    """Find the first evaluable, previously unused split in one seed window."""
    for seed in range(requested_seed, requested_seed + max_attempts):
        raw_folds = stratified_scaffold_group_kfold(
            prepared.smiles,
            prepared.y,
            k=outer_k,
            seed=seed,
        )
        folds = tuple(
            (
                np.asarray(train_idx, dtype=int),
                np.asarray(test_idx, dtype=int),
            )
            for train_idx, test_idx in raw_folds
        )
        try:
            _validate_partition(folds, prepared.y, outer_k, min_class_count)
        except ValueError:
            continue
        signature = _fold_partition_signature(folds, prepared.ik)
        if signature in used_signatures:
            continue
        return RepeatSplit(
            repeat_id=repeat_id,
            requested_seed=requested_seed,
            actual_seed=seed,
            signature=signature,
            folds=folds,
        )
    raise ValueError(
        f"No unique evaluable {outer_k}-fold scaffold split found for repeat "
        f"{repeat_id} in seed window [{requested_seed}, "
        f"{requested_seed + max_attempts - 1}]"
    )


def prepare_repeated_sets(
    variants: list[str],
    tasks: list[str],
    cells: list[str],
    offsets: pd.DataFrame,
    outer_seeds: tuple[int, ...],
    seed_search_attempts: int,
    cohort_dir: Path = calibrated.COHORT_DIR,
    gene_features_dir: Path = calibrated.GENE_FEATURES_DIR,
) -> tuple[list[RepeatedPreparedSet], list[dict[str, Any]]]:
    repeated_sets: list[RepeatedPreparedSet] = []
    split_rows: list[dict[str, Any]] = []
    anchor_variant = variants[0]

    for cell in cells:
        for task in tasks:
            variant_bases = {
                variant: calibrated.prepare_task_cell(
                    task,
                    cell,
                    variant,
                    offsets,
                    outer_seed=outer_seeds[0],
                    cohort_dir=cohort_dir,
                    gene_features_dir=gene_features_dir,
                )
                for variant in variants
            }
            anchor_base = variant_bases[anchor_variant]
            for variant, candidate in variant_bases.items():
                if (
                    candidate.ik != anchor_base.ik
                    or candidate.smiles != anchor_base.smiles
                    or not np.array_equal(candidate.y, anchor_base.y)
                ):
                    raise ValueError(
                        f"Preprocessing variant {variant} does not share the "
                        f"same cohort order for {task}/{cell}"
                    )

            used_signatures: set[str] = set()
            repeats: list[RepeatSplit] = []
            for repeat_id, requested_seed in enumerate(outer_seeds):
                split = find_unique_evaluable_repeat(
                    anchor_base,
                    repeat_id,
                    requested_seed,
                    used_signatures,
                    max_attempts=seed_search_attempts,
                )
                used_signatures.add(split.signature)
                repeats.append(split)
                split_rows.append(
                    {
                        "task": task,
                        "cell": cell,
                        "repeat_id": repeat_id,
                        "outer_seed_requested": requested_seed,
                        "outer_split_seed": split.actual_seed,
                        "outer_split_signature": split.signature,
                        "n_compounds": len(anchor_base.y),
                        "n_positive": int(np.sum(anchor_base.y == 1)),
                        "n_negative": int(np.sum(anchor_base.y == 0)),
                    }
                )

            for variant in variants:
                for split in repeats:
                    prepared = replace(
                        variant_bases[variant],
                        outer_folds=split.folds,
                        outer_seed_used=split.actual_seed,
                    )
                    repeated_sets.append(
                        RepeatedPreparedSet(repeat=split, prepared=prepared)
                    )
    return repeated_sets, split_rows


def _attach_repeat_metadata(
    repeat: RepeatSplit,
    outcome: calibrated.OuterFoldOutcome,
) -> RepeatedOutcome:
    metadata = {
        "repeat_id": repeat.repeat_id,
        "outer_fold_id": outcome.fold_id,
        "outer_seed_requested": repeat.requested_seed,
        "outer_split_seed": repeat.actual_seed,
        "outer_split_signature": repeat.signature,
    }
    if outcome.fold_row is not None:
        outcome.fold_row.update(metadata)
    for row in outcome.oof_rows:
        row.update(metadata)
    for row in outcome.alpha_rows:
        row.update(metadata)
    return RepeatedOutcome(repeat=repeat, outcome=outcome)


def _write_repeated_group_outputs(
    records: list[RepeatedOutcome],
    output_dir: Path,
    variant: str,
    task: str,
    cell: str,
    n_repeats: int,
) -> dict[str, Any] | None:
    group = sorted(
        [
            record
            for record in records
            if record.outcome.variant == variant
            and record.outcome.task == task
            and record.outcome.cell == cell
        ],
        key=lambda record: (record.repeat.repeat_id, record.outcome.fold_id),
    )
    successful = [record for record in group if record.outcome.success]
    if not successful:
        return None

    fold_df = pd.DataFrame(
        [record.outcome.fold_row for record in successful]
    ).sort_values(["repeat_id", "fold_id"])
    oof_df = pd.DataFrame(
        [
            row
            for record in successful
            for row in record.outcome.oof_rows
        ]
    ).sort_values(["repeat_id", "fold_id", "ik"])

    expected_draws = n_repeats * calibrated.OUTER_K
    if len(fold_df) == expected_draws:
        counts = fold_df.groupby("repeat_id")["fold_id"].nunique()
        if len(counts) != n_repeats or not np.all(counts == calibrated.OUTER_K):
            raise RuntimeError(
                f"Incomplete repeat-fold matrix for {variant}/{task}/{cell}"
            )
        oof_counts = oof_df.groupby(["repeat_id", "ik"]).size()
        if not np.all(oof_counts == 1):
            raise RuntimeError(
                f"OOF predictions are not unique per repeat/compound for "
                f"{variant}/{task}/{cell}"
            )
        compounds_per_repeat = oof_df.groupby("repeat_id")["ik"].nunique()
        if compounds_per_repeat.nunique() != 1:
            raise RuntimeError(
                f"Repeats do not cover the same compound cohort for "
                f"{variant}/{task}/{cell}"
            )

    beta_rows: list[dict[str, Any]] = []
    for record in successful:
        outcome = record.outcome
        for gene, beta in zip(outcome.gene_names, outcome.beta):
            beta_rows.append(
                {
                    "task": task,
                    "cell": cell,
                    "variant": variant,
                    "repeat_id": record.repeat.repeat_id,
                    "fold_id": outcome.fold_id,
                    "outer_fold_id": outcome.fold_id,
                    "outer_seed_requested": record.repeat.requested_seed,
                    "outer_split_seed": record.repeat.actual_seed,
                    "outer_split_signature": record.repeat.signature,
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
        [
            row
            for record in group
            for row in record.outcome.alpha_rows
        ]
    )

    stem = f"{variant}_{task}__{cell}"
    fold_df.to_csv(output_dir / f"repeated_nested_cv_{stem}.csv", index=False)
    oof_df.to_csv(
        output_dir / f"repeated_nested_cv_oof_predictions_{stem}.csv",
        index=False,
    )
    pd.DataFrame(beta_rows).to_csv(
        output_dir / f"repeated_nested_cv_beta_{stem}.csv",
        index=False,
    )
    if not alpha_df.empty:
        alpha_df.sort_values(
            ["repeat_id", "outer_fold", "inner_fold", "alpha"],
            ascending=[True, True, True, False],
        ).to_csv(
            output_dir / f"repeated_nested_cv_alpha_diagnostics_{stem}.csv",
            index=False,
        )

    summary: dict[str, Any] = {
        "task": task,
        "cell": cell,
        "variant": variant,
        "solver": str(fold_df["solver"].iloc[0]),
        "n_repeats": int(fold_df["repeat_id"].nunique()),
        "outer_folds_per_repeat": calibrated.OUTER_K,
        "n_paired_outer_draws": len(fold_df),
        "n_failed_outer_draws": len(group) - len(successful),
        "n_unique_compounds": int(oof_df["ik"].nunique()),
        "n_test_evaluations": int(len(oof_df)),
        "mean_delta_auprc": float(fold_df["delta_auprc"].mean()),
        "sd_delta_auprc": float(fold_df["delta_auprc"].std(ddof=1)),
        "mean_delta_auroc": float(fold_df["delta_auroc"].mean()),
        "sd_delta_auroc": float(fold_df["delta_auroc"].std(ddof=1)),
        "n_alpha_boundary_hits": int(fold_df["alpha_boundary_hit"].sum()),
        "n_folds_saturated_no_bio_signal": int(
            fold_df["beta_saturated_no_bio_signal"].sum()
        ),
    }
    if len(fold_df) >= 2:
        for metric in ("auprc", "auroc"):
            nb = repeated_cv_nadeau_bengio_test(
                fold_df[f"delta_{metric}"].to_numpy(dtype=float),
                fold_df["n_outer_train"].to_numpy(dtype=float),
                fold_df["n_outer_test"].to_numpy(dtype=float),
            )
            for key, value in nb.items():
                summary[f"nb_{metric}_{key}"] = value
    pd.DataFrame([summary]).to_csv(
        output_dir / f"repeated_nested_cv_summary_{stem}.csv",
        index=False,
    )
    return summary


def run_repeated_analysis(
    output_dir: Path,
    variants: list[str],
    tasks: list[str],
    cells: list[str],
    *,
    chem_offset_path: Path = calibrated.CHEM_OFFSET_PATH,
    cohort_dir: Path = calibrated.COHORT_DIR,
    gene_features_dir: Path = calibrated.GENE_FEATURES_DIR,
    deployment_report_path: Path | None = None,
    outer_seeds: tuple[int, ...] = DEFAULT_OUTER_SEEDS,
    seed_search_attempts: int = DEFAULT_SEED_SEARCH_ATTEMPTS,
    solver: str = "lbfgs",
    n_jobs: int = -1,
    points_per_decade: float = calibrated.ALPHA_POINTS_PER_DECADE,
    use_sample_weight: bool = False,
    maxiter: int = 500,
    retry_maxiter: int = 2000,
    gtol: float = 1e-6,
    acceptance_gradient: float = 1e-5,
    ftol: float = 1e-14,
) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite or resume existing output directory: "
            f"{output_dir}"
        )
    outer_seeds = parse_outer_seeds(outer_seeds)
    validate_seed_windows(outer_seeds, seed_search_attempts)
    output_dir.mkdir(parents=True, exist_ok=False)

    cohort_dir = cohort_dir.resolve()
    gene_features_dir = gene_features_dir.resolve()
    deployment_report_path = (
        deployment_report_path.resolve()
        if deployment_report_path is not None
        else chem_offset_path.resolve().parent
        / "chem_prior_deployment_report.csv"
    )
    offsets = calibrated.validate_chem_offsets(
        calibrated.load_chem_offsets(chem_offset_path),
        tasks,
    )
    _validate_stage1_deployment_report(deployment_report_path, tasks)
    chemical_offset_sha256 = _file_sha256(chem_offset_path)
    deployment_report_sha256 = _file_sha256(deployment_report_path)
    gene_feature_checksums = _gene_feature_checksums(
        gene_features_dir,
        cells,
        variants,
    )
    alphas = calibrated.alpha_grid(points_per_decade)
    repeated_sets, split_rows = prepare_repeated_sets(
        variants,
        tasks,
        cells,
        offsets,
        outer_seeds,
        seed_search_attempts,
        cohort_dir,
        gene_features_dir,
    )
    split_df = pd.DataFrame(split_rows)
    if split_df.duplicated(["task", "cell", "outer_split_signature"]).any():
        raise RuntimeError("Duplicate outer scaffold partitions survived validation")
    split_df.to_csv(output_dir / "repeated_outer_split_registry.csv", index=False)
    actual_seed_rows = [
        {
            "task": str(row["task"]),
            "cell": str(row["cell"]),
            "repeat_id": int(row["repeat_id"]),
            "outer_seed_requested": int(row["outer_seed_requested"]),
            "outer_split_seed": int(row["outer_split_seed"]),
            "outer_split_signature": str(row["outer_split_signature"]),
        }
        for row in sorted(
            split_rows,
            key=lambda item: (item["task"], item["cell"], item["repeat_id"]),
        )
    ]
    actual_seed_sets_by_task_cell: dict[str, set[int]] = {}
    for row in actual_seed_rows:
        key = f"{row['task']}__{row['cell']}"
        actual_seed_sets_by_task_cell.setdefault(key, set()).add(
            row["outer_split_seed"]
        )
    actual_seed_count_by_task_cell = {
        key: len(seeds)
        for key, seeds in sorted(actual_seed_sets_by_task_cell.items())
    }

    repeated_jobs = [
        RepeatedJob(
            repeat=repeated.repeat,
            job=calibrated.OuterFoldJob(
                prepared=repeated.prepared,
                fold_id=fold_id,
                outer_train_idx=train_idx,
                outer_test_idx=test_idx,
                solver=solver,
                use_sample_weight=use_sample_weight,
                alpha_grid=alphas,
                inner_cv_k=calibrated.INNER_CV_K,
                inner_cv_seed=calibrated.INNER_CV_SEED,
                maxiter=maxiter,
                retry_maxiter=retry_maxiter,
                gtol=gtol,
                acceptance_gradient=acceptance_gradient,
                ftol=ftol,
            ),
        )
        for repeated in repeated_sets
        for fold_id, (train_idx, test_idx) in enumerate(
            repeated.prepared.outer_folds
        )
    ]
    requested_groups = [
        (variant, task, cell)
        for variant in variants
        for cell in cells
        for task in tasks
    ]
    expected_jobs = (
        len(requested_groups) * len(outer_seeds) * calibrated.OUTER_K
    )
    resolved_jobs = min(
        max(1, effective_n_jobs(n_jobs)),
        max(1, len(repeated_jobs)),
    )
    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": PIPELINE_NAME,
        "stage2_model_spec": calibrated.STAGE2_MODEL_SPEC,
        "solver": solver,
        "chemical_offset_path": str(chem_offset_path.resolve()),
        "chemical_offset_sha256": chemical_offset_sha256,
        "chemical_offset_frozen": True,
        "chemical_offset_coefficient": 1.0,
        "cohort_dir": str(cohort_dir),
        "gene_features_dir": str(gene_features_dir),
        "gene_feature_files_sha256": gene_feature_checksums,
        "stage1_ensemble_rule": "task-specific top-4 candidate arithmetic mean",
        "stage1_k": 4,
        "stage1_deployment_report": str(deployment_report_path),
        "stage1_deployment_report_sha256": deployment_report_sha256,
        "intercept_penalized": False,
        "stage2_likelihood": (
            "weighted_sensitivity"
            if use_sample_weight
            else "unweighted_publication_primary"
        ),
        "variants": variants,
        "tasks": tasks,
        "cells": cells,
        "n_repeats": len(outer_seeds),
        "outer_folds_per_repeat": calibrated.OUTER_K,
        "outer_seed_requests": list(outer_seeds),
        "outer_seed_request_count": len(outer_seeds),
        "outer_split_actual_seeds": actual_seed_rows,
        "outer_split_actual_seed_count_by_task_cell": actual_seed_count_by_task_cell,
        "outer_split_registry_rows": len(split_df),
        "outer_seed_search_attempts": seed_search_attempts,
        "outer_seed_windows_nonoverlapping": True,
        "duplicate_outer_partitions_allowed": False,
        "outer_min_class_per_partition": calibrated.MIN_CLASS_PER_OUTER_PARTITION,
        "inner_folds": calibrated.INNER_CV_K,
        "inner_seed": calibrated.INNER_CV_SEED,
        "alpha_grid": list(alphas),
        "alpha_selection_scope": (
            "independently_within_each_repeat_outer_training_partition"
        ),
        "alpha_reused_across_outer_partitions": False,
        "cv_design": "fully_repeated_nested_scaffold_cv",
        "primary_inference": "nadeau_bengio_corrected_resampled_paired_t_test",
        "paired_comparison": "BioTox minus Chem-only on identical outer-test compounds",
        "multiple_testing": "Benjamini-Hochberg within variant and cell across 12 tasks",
        "confirmatory_multiplicity_family": (
            "standardized variant: 12 tasks within each cell; "
            "residualized variant reported as a separate sensitivity family"
        ),
        "requested_n_jobs": n_jobs,
        "resolved_n_jobs": resolved_jobs,
        "available_logical_cpus": os.cpu_count(),
        "parallel_unit": "repeat_outer_fold",
        "backend": "loky",
        "inner_max_num_threads": 1,
        "n_requested_groups": len(requested_groups),
        "n_expected_outer_fold_jobs": expected_jobs,
        "n_launched_outer_fold_jobs": len(repeated_jobs),
        "analysis_started": True,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[repeated-calibrated-v1] {len(repeated_jobs)}/{expected_jobs} "
        f"outer-fold jobs prepared; repeats={len(outer_seeds)}; "
        f"workers={resolved_jobs}; solver={solver}"
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
        )(
            delayed(calibrated.run_outer_fold)(repeated_job.job)
            for repeated_job in repeated_jobs
        )
    records = [
        _attach_repeat_metadata(repeated_job.repeat, outcome)
        for repeated_job, outcome in zip(repeated_jobs, outcomes)
    ]

    failures = [
        {
            "task": record.outcome.task,
            "cell": record.outcome.cell,
            "variant": record.outcome.variant,
            "repeat_id": record.repeat.repeat_id,
            "fold_id": record.outcome.fold_id,
            "outer_seed_requested": record.repeat.requested_seed,
            "outer_split_seed": record.repeat.actual_seed,
            "outer_split_signature": record.repeat.signature,
            "error": record.outcome.error,
        }
        for record in records
        if not record.outcome.success
    ]
    summaries = []
    for variant, task, cell in requested_groups:
        summary = _write_repeated_group_outputs(
            records,
            output_dir,
            variant,
            task,
            cell,
            len(outer_seeds),
        )
        if summary is not None:
            summaries.append(summary)
    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "repeated_outer_fold_failures.csv",
            index=False,
        )
    pd.DataFrame(summaries).to_csv(
        output_dir / "repeated_nested_cv_summary_all.csv",
        index=False,
    )

    successful_count = int(sum(record.outcome.success for record in records))
    completed_groups = int(
        sum(
            summary["n_paired_outer_draws"] == len(outer_seeds) * calibrated.OUTER_K
            for summary in summaries
        )
    )
    complete = bool(
        not failures
        and len(records) == expected_jobs
        and successful_count == expected_jobs
        and len(summaries) == len(requested_groups)
        and completed_groups == len(requested_groups)
    )
    manifest.update(
        {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_successful_outer_folds": successful_count,
            "n_failed_outer_folds": len(failures),
            "n_completed_groups": completed_groups,
            "analysis_complete": complete,
        }
    )
    completion_name = (
        "RUN_COMPLETE.json" if complete else "RUN_INCOMPLETE.json"
    )
    (output_dir / completion_name).write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if not complete:
        raise RuntimeError(
            f"{len(failures)} repeated outer-fold jobs failed; see "
            f"{output_dir / 'repeated_outer_fold_failures.csv'}"
        )
    return summaries
