#!/usr/bin/env python3
"""Audit a fresh seed-1 GPU refit against the frozen Stage-1 reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import stage1.splitter  # noqa: E402,F401 -- legacy pickle module alias


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def candidate_metrics(cache: dict) -> dict[str, float]:
    return {row["candidate"]: float(row["test_auprc"]) for row in cache["candidate_rows"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--refit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite audit report: {args.output}")
    reference_cache_dir = args.reference_root / "candidates" / "task_benchmark_cache"
    refit_cache_dir = args.refit_root / "candidates" / "task_benchmark_cache"
    reference_files = sorted(reference_cache_dir.glob("*.pkl"))
    refit_files = sorted(refit_cache_dir.glob("*.pkl"))
    if [path.name for path in reference_files] != [path.name for path in refit_files]:
        raise AssertionError("Reference and refit task-cache file sets differ")

    task_results = {}
    structural_pass = True
    for ref_path, new_path in zip(reference_files, refit_files):
        reference = load_pickle(ref_path)
        refit = load_pickle(new_path)
        same_contract = all(
            (
                reference.get("task") == refit.get("task"),
                int(reference.get("seed", -1)) == int(refit.get("seed", -1)) == 1,
                reference.get("candidate_names") == refit.get("candidate_names"),
                reference.get("ensemble_rule") == refit.get("ensemble_rule") == "mean",
                len(reference["y_test"]) == len(refit["y_test"]),
                int(np.sum(reference["y_test"])) == int(np.sum(refit["y_test"])),
                reference["cohort_ik"] == refit["cohort_ik"],
            )
        )
        structural_pass &= same_contract
        reference_metrics = candidate_metrics(reference)
        refit_metrics = candidate_metrics(refit)
        metric_differences = {
            name: refit_metrics[name] - reference_metrics[name]
            for name in reference_metrics
        }
        task_results[ref_path.stem] = {
            "structural_contract_match": same_contract,
            "seed": 1,
            "test_size": len(refit["y_test"]),
            "test_positive_count": int(np.sum(refit["y_test"])),
            "test_row_order_identical": bool(
                np.array_equal(reference["y_test"], refit["y_test"])
            ),
            "test_row_ids_available_in_cache": False,
            "candidate_auprc_delta_refit_minus_reference": metric_differences,
            "candidate_auprc_mean_absolute_delta": float(
                np.mean(np.abs(list(metric_differences.values())))
            ),
            "candidate_auprc_max_absolute_delta": float(
                np.max(np.abs(list(metric_differences.values())))
            ),
            "twelve_candidate_mean_auprc_reference": float(
                reference["test_ensemble_auprc"]
            ),
            "twelve_candidate_mean_auprc_refit": float(refit["test_ensemble_auprc"]),
            "reference_sha256": sha256(ref_path),
            "refit_sha256": sha256(new_path),
        }

    reference_selection = pd.read_csv(
        args.reference_root / "topk_selection" / "stage1_topk_k4_reference_by_task.csv"
    ).set_index("task")
    refit_selection = pd.read_csv(
        args.refit_root / "topk_selection" / "stage1_topk_k4_reference_by_task.csv"
    ).set_index("task")
    selection_results = {}
    for task in reference_selection.index:
        reference_top4 = reference_selection.loc[task, "ranked_candidates"].split(";")[:4]
        refit_top4 = refit_selection.loc[task, "ranked_candidates"].split(";")[:4]
        selection_results[task] = {
            "reference_top4": reference_top4,
            "refit_top4": refit_top4,
            "top4_overlap_count": len(set(reference_top4) & set(refit_top4)),
            "k4_auprc_reference": float(reference_selection.loc[task, "k4_auprc"]),
            "k4_auprc_refit": float(refit_selection.loc[task, "k4_auprc"]),
            "k4_auprc_delta_refit_minus_reference": float(
                refit_selection.loc[task, "k4_auprc"]
                - reference_selection.loc[task, "k4_auprc"]
            ),
        }

    reference_offset = pd.read_csv(
        args.reference_root / "deployed_k4" / "chem_offset_final.csv"
    ).set_index("ik").sort_index()
    refit_offset = pd.read_csv(
        args.refit_root / "deployed_k4" / "chem_offset_final.csv"
    ).set_index("ik").sort_index()
    if not reference_offset.index.equals(refit_offset.index):
        raise AssertionError("Reference and refit deployment compound sets differ")
    if list(reference_offset.columns) != list(refit_offset.columns):
        raise AssertionError("Reference and refit deployment task columns differ")
    offset_results = {}
    for column in reference_offset.columns:
        left = reference_offset[column].to_numpy(dtype=float)
        right = refit_offset[column].to_numpy(dtype=float)
        offset_results[column] = {
            "pearson_r": float(np.corrcoef(left, right)[0, 1]),
            "mean_absolute_delta": float(np.mean(np.abs(right - left))),
            "max_absolute_delta": float(np.max(np.abs(right - left))),
        }

    report = {
        "status": (
            "FAIL_NUMERICAL_REPRODUCIBILITY"
            if structural_pass
            else "FAIL_STRUCTURAL_AND_NUMERICAL_REPRODUCIBILITY"
        ),
        "structural_status": "PASS" if structural_pass else "FAIL",
        "numerical_status": "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_seed": 1,
        "model_refit_performed": True,
        "candidate_cache_reused": False,
        "existing_results_overwritten": False,
        "numerically_identical_to_frozen_reference": False,
        "interpretation": (
            "The refit reproduces the frozen cohort, seed, endpoint set, candidate "
            "library, and K=4 deployment procedure. Numerical identity is not claimed. "
            "The cache schema stores test labels and predictions without test-row IDs; "
            "the observed test-row order differs, so elementwise prediction comparison "
            "would be invalid. Candidate AUPRC, Top-4 membership, and deployed offsets "
            "are reported instead. Because the deployed offsets differ from the frozen "
            "reference, this audit does not classify the refit as numerically reproducible."
        ),
        "reference_root": str(args.reference_root.resolve()),
        "refit_root": str(args.refit_root.resolve()),
        "task_caches": task_results,
        "k4_selection": selection_results,
        "deployed_offset": offset_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"])


if __name__ == "__main__":
    main()
