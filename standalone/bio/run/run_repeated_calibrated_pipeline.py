"""Non-overwriting repeated calibrated Stage-2 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
_STANDALONE_ROOT = HERE.parents[1]


def _register_bundled_package(name: str, *locations: Path) -> None:
    """Make a bundled sibling ``run/`` directory importable under its
    original package name -- both in this process and in any subprocess
    or multiprocessing worker it spawns.

    An in-memory ``sys.modules`` entry alone is invisible to spawned
    worker processes (e.g. joblib/loky ridge-fitting workers): each
    starts a fresh interpreter that must resolve ``publication.X`` /
    ``chem.X`` imports itself when unpickling a task, using only the
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
    combine modules that physically live under both ``bio/run`` and
    ``chem/run``), and adds its PARENT directory to ``sys.path`` and to
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

    alias_root = Path(os.environ.get("BIOTOX_PACKAGE_ALIAS_ROOT", _STANDALONE_ROOT / "_pkg_alias"))
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


_register_bundled_package("chem", _STANDALONE_ROOT / "chem" / "run")
_register_bundled_package("publication", HERE, _STANDALONE_ROOT / "chem" / "run")

from publication.nested_cv_offset_logistic_calibrated_repeated import (  # noqa: E402
    DEFAULT_OUTER_SEEDS,
    DEFAULT_SEED_SEARCH_ATTEMPTS,
    parse_outer_seeds,
    run_repeated_analysis,
    validate_seed_windows,
)
from publication.nested_cv_offset_logistic_calibrated import (  # noqa: E402
    CELL_IDS,
    CHEM_OFFSET_PATH,
    COHORT_DIR,
    GENE_FEATURES_DIR,
    TOX21_TASKS,
)
from publication.summarize_repeated_cv_publication import summarize  # noqa: E402


DEFAULT_OUT_ROOT = (
    HERE
    / "_run_output"
    / "nested_cv_k4_calibrated_lbfgs_repeated20x5_fresh_v1"
)
def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv_list(value: str | None, allowed: list[str]) -> list[str]:
    if value is None:
        return list(allowed)
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = sorted(set(selected) - set(allowed))
    if invalid:
        raise ValueError(f"Unsupported values: {', '.join(invalid)}")
    return selected


def validate_new_output(path: Path) -> None:
    if "uncalibrated" in path.name.lower():
        raise ValueError(f"Unsafe repeated calibrated output name: {path}")
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite or resume existing output directory: {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run repeated calibrated nested scaffold CV and Nadeau-Bengio "
            "paired inference without modifying prior outputs."
        )
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--chem-offset-path", type=Path, default=CHEM_OFFSET_PATH)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    parser.add_argument(
        "--gene-features-dir",
        type=Path,
        default=GENE_FEATURES_DIR,
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
    parser.add_argument("--cells", help="Comma-separated cell subset")
    parser.add_argument(
        "--outer-seeds",
        default=",".join(str(seed) for seed in DEFAULT_OUTER_SEEDS),
        help=(
            "Comma-separated predeclared outer split seeds. Defaults to twenty "
            "non-overlapping 1000-seed search windows."
        ),
    )
    parser.add_argument(
        "--seed-search-attempts",
        type=int,
        default=DEFAULT_SEED_SEARCH_ATTEMPTS,
    )
    parser.add_argument(
        "--ridge-solver",
        choices=["lbfgs", "statsmodels"],
        default="lbfgs",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--points-per-decade", type=float, default=2.0)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--retry-maxiter", type=int, default=2000)
    parser.add_argument("--gtol", type=float, default=1e-6)
    parser.add_argument("--acceptance-gradient", type=float, default=1e-5)
    parser.add_argument("--ftol", type=float, default=1e-14)
    parser.add_argument("--use-sample-weight", action="store_true")
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Run repeated CV only; do not generate the NB summary afterward.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-posthoc", action="store_true", help="Skip gene means, compound tests, and held-out gene attributions.")
    args = parser.parse_args()

    output_root = args.out_root.resolve()
    summary_root = output_root.with_name(output_root.name + "_summary")
    posthoc_root = output_root.with_name(output_root.name + "_posthoc")
    chemical_offset = args.chem_offset_path.resolve()
    cohort_dir = args.cohort_dir.resolve()
    gene_features_dir = args.gene_features_dir.resolve()
    deployment_report_path = (
        chemical_offset.parent / "chem_prior_deployment_report.csv"
    )
    validate_new_output(output_root)
    if not args.skip_summary:
        validate_new_output(summary_root)
    if not args.skip_posthoc:
        validate_new_output(posthoc_root)
    if not chemical_offset.is_file():
        raise FileNotFoundError(chemical_offset)

    variants = (
        ["standardized", "residualized"]
        if args.variant == "both"
        else [args.variant]
    )
    tasks = parse_csv_list(args.tasks, TOX21_TASKS)
    cells = parse_csv_list(args.cells, CELL_IDS)
    outer_seeds = parse_outer_seeds(args.outer_seeds)
    validate_seed_windows(outer_seeds, args.seed_search_attempts)
    checksum_before = file_sha256(chemical_offset)
    if not deployment_report_path.is_file():
        raise FileNotFoundError(deployment_report_path)
    deployment_checksum = file_sha256(deployment_report_path)
    specification = {
        "output_root": str(output_root),
        "summary_root": None if args.skip_summary else str(summary_root),
        "chemical_offset_path": str(chemical_offset),
        "chemical_offset_sha256": checksum_before,
        "chemical_offset_frozen": True,
        "chemical_offset_coefficient": 1.0,
        "cohort_dir": str(cohort_dir),
        "gene_features_dir": str(gene_features_dir),
        "stage1_k4_deployment_report": str(deployment_report_path),
        "stage1_k4_deployment_report_sha256": deployment_checksum,
        "variants": variants,
        "tasks": tasks,
        "cells": cells,
        "outer_seeds": list(outer_seeds),
        "n_repeats": len(outer_seeds),
        "outer_folds_per_repeat": 5,
        "paired_outer_draws_per_task_cell": len(outer_seeds) * 5,
        "inner_folds_per_outer_partition": 3,
        "alpha_selection_scope": (
            "independently_within_each_repeat_outer_training_partition"
        ),
        "alpha_reused_across_outer_partitions": False,
        "cv_design": "fully_repeated_nested_scaffold_cv",
        "primary_inference": "Nadeau-Bengio corrected resampled paired t-test",
        "requested_n_jobs": args.n_jobs,
        "available_logical_cpus": os.cpu_count(),
        "stage2_likelihood": (
            "weighted_sensitivity"
            if args.use_sample_weight
            else "unweighted_publication_primary"
        ),
        "intercept_penalized": False,
        "calibration_intercept_frozen": True,
        "calibration_policy": "fit_on_each_training_subset_then_freeze_during_beta_fit",
    }
    print(json.dumps(specification, indent=2))
    if args.dry_run:
        print("[repeated-calibrated-v1] dry run complete; no output created.")
        return

    run_repeated_analysis(
        output_dir=output_root,
        variants=variants,
        tasks=tasks,
        cells=cells,
        chem_offset_path=chemical_offset,
        cohort_dir=cohort_dir,
        gene_features_dir=gene_features_dir,
        deployment_report_path=deployment_report_path,
        outer_seeds=outer_seeds,
        seed_search_attempts=args.seed_search_attempts,
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
    checksum_after = file_sha256(chemical_offset)
    if checksum_after != checksum_before:
        raise RuntimeError("Frozen Stage-1 chemical offset changed during Stage 2")
    if file_sha256(deployment_report_path) != deployment_checksum:
        raise RuntimeError("Frozen Stage-1 K=4 deployment report changed")
    if not args.skip_summary:
        summarize(output_root, summary_root)
    if not args.skip_posthoc:
        from methods_residual_posthoc import run as run_posthoc
        run_posthoc(output_root, posthoc_root, expected_repeats=len(outer_seeds))
    print(f"[repeated-calibrated-v1] complete: {output_root}")
    if not args.skip_summary:
        print(f"[repeated-calibrated-v1] summary: {summary_root}")
    print(f"[repeated-calibrated-v1] frozen offset checksum: {checksum_after}")


if __name__ == "__main__":
    main()
