#!/usr/bin/env python3
"""Compare a new seed-1 Stage-1 assembly with the frozen reference outputs."""
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

import stage1.splitter  # noqa: E402,F401 -- registers the legacy pickle alias


TASKS = (
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cache(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def compare_cache(reference: Path, reproduced: Path) -> dict:
    left = load_cache(reference)
    right = load_cache(reproduced)
    if int(left.get("seed", -1)) != 1 or int(right.get("seed", -1)) != 1:
        raise AssertionError(f"seed mismatch: {reference} vs {reproduced}")
    for key in ("task", "candidate_names", "ensemble_rule"):
        if left.get(key) != right.get(key):
            raise AssertionError(f"{key} mismatch: {reference} vs {reproduced}")
    for key in ("y_test", "cohort_ik"):
        np.testing.assert_array_equal(np.asarray(left[key]), np.asarray(right[key]))
    for key in ("test_preds_by_candidate", "cohort_preds_by_candidate"):
        if set(left[key]) != set(right[key]):
            raise AssertionError(f"{key} candidates differ for {left['task']}")
        for candidate in left[key]:
            np.testing.assert_array_equal(
                np.asarray(left[key][candidate]), np.asarray(right[key][candidate])
            )
    return {
        "reference_sha256": sha256(reference),
        "reproduced_sha256": sha256(reproduced),
        "semantic_match": True,
        "seed": 1,
        "candidate_count": len(left["candidate_names"]),
    }


def compare_csv(reference: Path, reproduced: Path) -> dict:
    left = pd.read_csv(reference)
    right = pd.read_csv(reproduced)
    pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-14, rtol=1e-14)
    return {
        "reference_sha256": sha256(reference),
        "reproduced_sha256": sha256(reproduced),
        "table_match": True,
        "rows": len(left),
        "columns": len(left.columns),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-candidates", required=True, type=Path)
    parser.add_argument("--reference-selection", required=True, type=Path)
    parser.add_argument("--reference-deployment", required=True, type=Path)
    parser.add_argument("--reproduced-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite check report: {args.output}")
    candidate_results = {}
    for task in TASKS:
        name = f"{task}.pkl"
        candidate_results[name] = compare_cache(
            args.reference_candidates / "task_benchmark_cache" / name,
            args.reproduced_root / "candidates" / "task_benchmark_cache" / name,
        )
    csv_results = {}
    comparisons = (
        (
            "selection/stage1_topk_ensemble_curve_revised.csv",
            args.reference_selection / "stage1_topk_ensemble_curve_revised.csv",
            args.reproduced_root / "topk_selection" / "stage1_topk_ensemble_curve_revised.csv",
        ),
        (
            "selection/stage1_topk_k4_reference_by_task.csv",
            args.reference_selection / "stage1_topk_k4_reference_by_task.csv",
            args.reproduced_root / "topk_selection" / "stage1_topk_k4_reference_by_task.csv",
        ),
        (
            "selection/stage1_topk_macro_summary_revised.csv",
            args.reference_selection / "stage1_topk_macro_summary_revised.csv",
            args.reproduced_root / "topk_selection" / "stage1_topk_macro_summary_revised.csv",
        ),
        (
            "deployment/chem_offset_final.csv",
            args.reference_deployment / "chem_offset_final.csv",
            args.reproduced_root / "deployed_k4" / "chem_offset_final.csv",
        ),
        (
            "deployment/chem_prior_deployment_report.csv",
            args.reference_deployment / "chem_prior_deployment_report.csv",
            args.reproduced_root / "deployed_k4" / "chem_prior_deployment_report.csv",
        ),
    )
    for label, reference, reproduced in comparisons:
        csv_results[label] = compare_csv(reference, reproduced)
    report = {
        "status": "PASS",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_seed": 1,
        "candidate_cache_mode": "read_only_reference_cache_reassembly",
        "model_refit_performed": False,
        "existing_results_overwritten": False,
        "reference_candidates": str(args.reference_candidates.resolve()),
        "reference_selection": str(args.reference_selection.resolve()),
        "reference_deployment": str(args.reference_deployment.resolve()),
        "reproduced_root": str(args.reproduced_root.resolve()),
        "candidate_caches": candidate_results,
        "csv_outputs": csv_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: seed-1 Stage-1 candidate predictions and K=4 outputs match; {args.output}")


if __name__ == "__main__":
    main()
