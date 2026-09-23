"""Run paired-time BioTox with deterministic inner-seed recovery.

The version-1 repeated scaffold partitions are preserved exactly. For each
outer-training partition, this wrapper starts from the predeclared inner seed
42 and advances deterministically only when the resulting three-fold inner
split is not binary-evaluable. The fitted statistical model and frozen K=4
chemical prior are unchanged.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from publication import nested_cv_offset_logistic_calibrated_repeated_v1 as repeated
from publication import nested_cv_offset_logistic_calibrated_v2 as calibrated
from publication import run_repeated_calibrated_timepoint_v1 as timepoint_v1


PIPELINE_NAME = "calibrated_stage2_repeated_nested_cv_innerseed_v3"
INNER_SEED_SEARCH_ATTEMPTS = 1000
_ORIGINAL_RUN_OUTER_FOLD = calibrated.run_outer_fold


def find_binary_evaluable_inner_seed(
    job: calibrated.OuterFoldJob,
    *,
    max_attempts: int = INNER_SEED_SEARCH_ATTEMPTS,
) -> int:
    """Return the first inner seed yielding three binary-evaluable folds."""
    outer_train_idx = np.asarray(job.outer_train_idx, dtype=int)
    y_train = job.prepared.y[outer_train_idx]
    smiles_train = [
        job.prepared.smiles[int(index)] for index in outer_train_idx
    ]
    requested_seed = int(job.inner_cv_seed)
    for seed in range(requested_seed, requested_seed + max_attempts):
        inner_folds = repeated.stratified_scaffold_group_kfold(
            smiles_train,
            y_train,
            k=job.inner_cv_k,
            seed=seed,
        )
        if len(inner_folds) != job.inner_cv_k:
            continue
        evaluable = True
        for inner_train_idx, inner_valid_idx in inner_folds:
            y_inner_train = y_train[np.asarray(inner_train_idx, dtype=int)]
            y_inner_valid = y_train[np.asarray(inner_valid_idx, dtype=int)]
            if (
                np.unique(y_inner_train).size != 2
                or np.unique(y_inner_valid).size != 2
            ):
                evaluable = False
                break
        if evaluable:
            return seed
    raise ValueError(
        "No binary-evaluable inner scaffold split found in seed window "
        f"[{requested_seed}, {requested_seed + max_attempts - 1}]"
    )


def run_outer_fold_with_inner_seed_search(
    job: calibrated.OuterFoldJob,
) -> calibrated.OuterFoldOutcome:
    requested_seed = int(job.inner_cv_seed)
    try:
        actual_seed = find_binary_evaluable_inner_seed(job)
    except Exception as exc:
        outcome = _ORIGINAL_RUN_OUTER_FOLD(job)
        outcome.success = False
        outcome.error = f"{type(exc).__name__}: {exc}"
        return outcome

    outcome = _ORIGINAL_RUN_OUTER_FOLD(
        replace(job, inner_cv_seed=actual_seed)
    )
    metadata = {
        "inner_cv_seed_requested": requested_seed,
        "inner_cv_seed_used": actual_seed,
        "inner_cv_seed_search_attempts": actual_seed - requested_seed + 1,
    }
    if outcome.fold_row is not None:
        outcome.fold_row.update(metadata)
    for row in outcome.oof_rows:
        row.update(metadata)
    for row in outcome.alpha_rows:
        row.update(metadata)
    return outcome


def install_inner_seed_search() -> None:
    repeated.PIPELINE_NAME = PIPELINE_NAME
    calibrated.run_outer_fold = run_outer_fold_with_inner_seed_search


def write_inner_seed_registry(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.glob("repeated_nested_cv_*.csv")):
        if not path.name.startswith(
            (
                "repeated_nested_cv_standardized_",
                "repeated_nested_cv_residualized_",
            )
        ):
            continue
        frame = pd.read_csv(path)
        required = {
            "task",
            "cell",
            "variant",
            "repeat_id",
            "outer_fold_id",
            "outer_split_seed",
            "outer_split_signature",
            "inner_cv_seed_requested",
            "inner_cv_seed_used",
            "inner_cv_seed_search_attempts",
        }
        if not required.issubset(frame.columns):
            continue
        rows.append(frame[list(sorted(required))])
    if not rows:
        raise RuntimeError("No completed outer-fold files contained inner seeds")

    registry = pd.concat(rows, ignore_index=True)
    expected_variant_rows = 2 * 3 * 12 * 20 * 5
    if len(registry) != expected_variant_rows:
        raise RuntimeError(
            f"Expected {expected_variant_rows} variant-specific inner-seed "
            f"rows, found {len(registry)}"
        )
    identity = ["task", "cell", "repeat_id", "outer_fold_id"]
    seed_counts = registry.groupby(identity, dropna=False)[
        "inner_cv_seed_used"
    ].nunique()
    if not (seed_counts == 1).all():
        raise RuntimeError(
            "Preprocessing variants used different inner seeds"
        )
    compact = (
        registry.sort_values(identity + ["variant"])
        .drop_duplicates(identity)
        .sort_values(identity)
        .reset_index(drop=True)
    )
    expected_outer_folds = 3 * 12 * 20 * 5
    if len(compact) != expected_outer_folds:
        raise RuntimeError(
            f"Expected {expected_outer_folds} inner-seed records, "
            f"found {len(compact)}"
        )
    compact.to_csv(
        output_root / "repeated_inner_split_registry.csv",
        index=False,
    )

    policy = {
        "pipeline": PIPELINE_NAME,
        "outer_partition_policy": (
            "unchanged from repeated calibrated v1"
        ),
        "inner_seed_requested": calibrated.INNER_CV_SEED,
        "inner_seed_search_attempts": INNER_SEED_SEARCH_ATTEMPTS,
        "inner_seed_policy": (
            "first seed at or above 42 producing three inner scaffold folds "
            "with both classes in train and validation"
        ),
        "n_outer_fold_records": len(compact),
        "n_inner_seed_adjustments": int(
            (compact["inner_cv_seed_used"] != calibrated.INNER_CV_SEED).sum()
        ),
        "chemical_prior_changed": False,
        "statistical_model_changed": False,
    }
    (output_root / "INNER_SEED_POLICY_V3.json").write_text(
        json.dumps(policy, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    install_inner_seed_search()
    timepoint_v1.main()

    args = timepoint_v1.parse_args()
    output_root = args.out_root.resolve()
    if output_root.is_dir() and not args.dry_run:
        write_inner_seed_registry(output_root)


if __name__ == "__main__":
    main()
