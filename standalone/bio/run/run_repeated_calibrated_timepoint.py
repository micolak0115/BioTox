"""Run the calibrated BioTox model on paired 6 h or 24 h inputs.

This wrapper leaves the finalized 6-hour pipeline untouched. It installs a
timepoint-specific cohort loader only for the current Python process, then
calls the audited repeated nested-CV implementation.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from publication import nested_cv_offset_logistic_calibrated as calibrated  # noqa: E402
from publication.nested_cv_offset_logistic_calibrated_repeated import (  # noqa: E402
    DEFAULT_OUTER_SEEDS,
    DEFAULT_SEED_SEARCH_ATTEMPTS,
    parse_outer_seeds,
    run_repeated_analysis,
    validate_seed_windows,
)
from publication.summarize_repeated_cv_publication import summarize  # noqa: E402
from publication.utils import load_matched_split  # noqa: E402


DEFAULT_INPUT_ROOT = (
    HERE / "_run_output" / "paired_timepoint_inputs_6h24h_8to12uM_v1"
)
CHEM_OFFSET_PATH = (
    HERE / "_run_output" / "chem_offset_k4_deployed" / "chem_offset_final.csv"
)
VARIANT_SUFFIX = {"standardized": "raw", "residualized": "residualized"}
DEFAULT_CELLS = ("HA1E", "HT29", "MCF7")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv_list(
    value: str | None,
    allowed: tuple[str, ...] | list[str],
    default: tuple[str, ...] | list[str],
) -> list[str]:
    if value is None:
        return list(default)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    if not selected:
        raise ValueError("At least one task/cell value is required")
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return selected


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(
    *,
    pert_time: int,
    cohort_dir: Path,
    gene_features_dir: Path,
    cells: list[str],
    variants: list[str],
) -> tuple[dict, dict]:
    cohort_manifest = load_manifest(cohort_dir / "COHORT_MANIFEST.json")
    feature_manifest = load_manifest(
        gene_features_dir / "FEATURE_MANIFEST.json"
    )
    expected_window = [8.0, 12.0]
    if cohort_manifest.get("dose_window_um") != expected_window:
        raise ValueError("Paired cohort concentration window is not 8-12 uM")
    if feature_manifest.get("dose_window_um") != expected_window:
        raise ValueError("Gene-feature concentration window is not 8-12 uM")
    if float(feature_manifest.get("feature_time_hours")) != float(pert_time):
        raise ValueError(
            "Gene-feature manifest time does not match --pert-time"
        )
    if feature_manifest.get("obsm_key") != "X_centered":
        raise ValueError("Only obsm['X_centered'] is valid for this analysis")
    if feature_manifest.get("paired_cohort_definition") != (
        "per-cell Tox21 compounds observed at both 6h and 24h"
    ):
        raise ValueError("Input is not the paired 6h/24h cohort")
    if Path(feature_manifest.get("cohort_dir", "")).resolve() != (
        cohort_dir.resolve()
    ):
        raise ValueError("Feature and cohort manifests reference different cohorts")

    for cell in cells:
        cell_record = cohort_manifest["cells"].get(cell)
        if cell_record is None:
            raise ValueError(f"Missing paired cohort metadata for {cell}")
        if cell_record.get("paired_stage1_scaffold_overlap") != 0:
            raise ValueError(f"{cell}: Stage-1 scaffold overlap is non-zero")
        cohort_path = (
            cohort_dir
            / f"tox21_scaffold_df_split_{cell}_8to12uM_paired6h24h.pkl"
        )
        if not cohort_path.is_file():
            raise FileNotFoundError(cohort_path)
        expected_cohort_hash = cell_record.get("cohort_file_sha256")
        if file_sha256(cohort_path) != expected_cohort_hash:
            raise ValueError(f"{cell}: paired cohort checksum mismatch")
        time_key = f"{pert_time}h"
        expected_feature_hashes = cell_record.get(
            "feature_files_sha256", {}
        ).get(time_key, {})
        for variant in variants:
            suffix = VARIANT_SUFFIX[variant]
            feature_path = (
                gene_features_dir / f"{cell}_gene_features_{suffix}.csv"
            )
            if not feature_path.is_file():
                raise FileNotFoundError(feature_path)
            expected_feature_hash = expected_feature_hashes.get(
                feature_path.name
            )
            if file_sha256(feature_path) != expected_feature_hash:
                raise ValueError(
                    f"{cell}: {feature_path.name} checksum mismatch"
                )
    return cohort_manifest, feature_manifest


def install_paired_cohort_loader() -> None:
    @functools.lru_cache(maxsize=None)
    def load_cell_data(
        cell: str,
        variant: str,
        cohort_dir: str,
        gene_features_dir: str,
    ):
        compounds = load_matched_split(
            str(
                Path(cohort_dir)
                / (
                    f"tox21_scaffold_df_split_{cell}_"
                    "8to12uM_paired6h24h.pkl"
                )
            )
        )
        suffix = VARIANT_SUFFIX[variant]
        features = pd.read_csv(
            Path(gene_features_dir)
            / f"{cell}_gene_features_{suffix}.csv"
        )
        return compounds, features

    calibrated._load_cell_data = load_cell_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe paired-time BioTox repeated nested CV at "
            "6 or 24 hours."
        )
    )
    parser.add_argument("--pert-time", type=int, choices=[6, 24], required=True)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument(
        "--chem-offset-path", type=Path, default=CHEM_OFFSET_PATH
    )
    # CEViChE-residualized variant is deprecated; unadjusted ("standardized") is
    # now the only supported mode. Original choice kept commented for provenance.
    # parser.add_argument(
    #     "--variant",
    #     choices=["standardized", "residualized", "both"],
    #     default="both",
    # )
    parser.add_argument(
        "--variant",
        choices=["standardized"],
        default="standardized",
    )
    parser.add_argument("--tasks", help="Comma-separated task subset")
    parser.add_argument(
        "--cells",
        help=(
            "Comma-separated cells. Default excludes HEPG2 because its paired "
            "cohort has only 46 compounds and cannot support all 12 five-fold "
            "analyses."
        ),
    )
    parser.add_argument(
        "--outer-seeds",
        default=",".join(str(seed) for seed in DEFAULT_OUTER_SEEDS),
    )
    parser.add_argument(
        "--seed-search-attempts",
        type=int,
        default=DEFAULT_SEED_SEARCH_ATTEMPTS,
    )
    parser.add_argument("--n-jobs", type=int, default=30)
    parser.add_argument("--points-per-decade", type=float, default=2.0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--retry-maxiter", type=int, default=2000)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--acceptance-gradient", type=float, default=1e-5)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--use-sample-weight", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-posthoc", action="store_true", help="Skip gene means, compound tests, and held-out gene attributions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    cohort_dir = input_root / "cohort"
    gene_features_dir = input_root / f"gene_features_{args.pert_time}h"
    output_root = args.out_root.resolve()
    summary_root = output_root.with_name(output_root.name + "_summary")
    posthoc_root = output_root.with_name(output_root.name + "_posthoc")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_root}")
    if not args.skip_summary and summary_root.exists():
        raise FileExistsError(f"Refusing to overwrite summary: {summary_root}")
    if not args.skip_posthoc and posthoc_root.exists():
        raise FileExistsError(f"Refusing to overwrite post-hoc outputs: {posthoc_root}")

    variants = (
        ["standardized", "residualized"]
        if args.variant == "both"
        else [args.variant]
    )
    tasks = parse_csv_list(
        args.tasks,
        list(calibrated.TOX21_TASKS),
        list(calibrated.TOX21_TASKS),
    )
    cells = parse_csv_list(
        args.cells,
        list(calibrated.CELL_IDS),
        list(DEFAULT_CELLS),
    )
    outer_seeds = parse_outer_seeds(args.outer_seeds)
    validate_seed_windows(outer_seeds, args.seed_search_attempts)
    cohort_manifest, feature_manifest = validate_inputs(
        pert_time=args.pert_time,
        cohort_dir=cohort_dir,
        gene_features_dir=gene_features_dir,
        cells=cells,
        variants=variants,
    )

    chemical_offset = args.chem_offset_path.resolve()
    deployment_report = (
        chemical_offset.parent / "chem_prior_deployment_report.csv"
    )
    if not chemical_offset.is_file():
        raise FileNotFoundError(chemical_offset)
    if not deployment_report.is_file():
        raise FileNotFoundError(deployment_report)
    offset_checksum = file_sha256(chemical_offset)
    deployment_checksum = file_sha256(deployment_report)

    specification = {
        "analysis": "paired_timepoint_sensitivity",
        "pert_time_hours": args.pert_time,
        "dose_window_um": [8.0, 12.0],
        "paired_compounds_required": True,
        "cohort_dir": str(cohort_dir),
        "gene_features_dir": str(gene_features_dir),
        "chemical_offset_path": str(chemical_offset),
        "chemical_offset_sha256": offset_checksum,
        "chemical_offset_frozen": True,
        "chemical_offset_coefficient": 1.0,
        "stage1_ensemble": "task-specific K=4 arithmetic mean",
        "variants": variants,
        "tasks": tasks,
        "cells": cells,
        "outer_seeds": list(outer_seeds),
        "n_repeats": len(outer_seeds),
        "outer_folds_per_repeat": 5,
        "n_jobs": args.n_jobs,
        "output_root": str(output_root),
        "summary_root": None if args.skip_summary else str(summary_root),
        "hepg2_complete_matrix_excluded": "HEPG2" not in cells,
        # HEPG2 may have been excluded further upstream, at cohort
        # construction (prepare_paired_timepoints.cells), not just from
        # this run's --cells selection -- in that case there is no
        # cohort_manifest["cells"]["HEPG2"] entry to report a paired
        # compound count from at all.
        "hepg2_paired_compounds": cohort_manifest["cells"]
        .get("HEPG2", {})
        .get("n_paired"),
        "input_manifest_feature_time": feature_manifest[
            "feature_time_hours"
        ],
    }
    print(json.dumps(specification, indent=2))
    if args.dry_run:
        print("[paired-time] dry run complete; no output created.")
        return

    install_paired_cohort_loader()
    run_repeated_analysis(
        output_dir=output_root,
        variants=variants,
        tasks=tasks,
        cells=cells,
        chem_offset_path=chemical_offset,
        cohort_dir=cohort_dir,
        gene_features_dir=gene_features_dir,
        deployment_report_path=deployment_report,
        outer_seeds=outer_seeds,
        seed_search_attempts=args.seed_search_attempts,
        solver="lbfgs",
        n_jobs=args.n_jobs,
        points_per_decade=args.points_per_decade,
        use_sample_weight=args.use_sample_weight,
        maxiter=args.maxiter,
        retry_maxiter=args.retry_maxiter,
        gtol=args.gtol,
        acceptance_gradient=args.acceptance_gradient,
        ftol=args.ftol,
    )
    if file_sha256(chemical_offset) != offset_checksum:
        raise RuntimeError("Frozen chemical offset changed during Stage 2")
    if file_sha256(deployment_report) != deployment_checksum:
        raise RuntimeError("Frozen K=4 deployment report changed during Stage 2")

    timepoint_manifest = {
        **specification,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "chemical_offset_sha256_after": file_sha256(chemical_offset),
        "deployment_report_sha256": deployment_checksum,
        "complete": True,
    }
    (output_root / "TIMEPOINT_EXPERIMENT_MANIFEST.json").write_text(
        json.dumps(timepoint_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if not args.skip_summary:
        summarize(output_root, summary_root)
    if not args.skip_posthoc:
        from methods_residual_posthoc import run as run_posthoc
        run_posthoc(output_root, posthoc_root, expected_repeats=len(outer_seeds))
    print(f"[paired-time] complete: {output_root}")
    if not args.skip_summary:
        print(f"[paired-time] summary: {summary_root}")


if __name__ == "__main__":
    main()
