#!/usr/bin/env python3
"""Config-driven standalone transcriptomic residual-learning workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parent
STAGE2_RUN = ROOT / "stage2"
STAGE1_RUN = ROOT / "stage1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2 import build_gene_features
from stage2 import prepare_paired_timepoint_inputs

WORKFLOW_ORDER = (
    "build_features",
    "prepare_paired_timepoints",
    "run_primary_residual_cv",
    "run_paired_6h_residual_cv",
    "run_paired_24h_residual_cv",
    "run_calibration_sensitivity",
)


def _prepare_runtime() -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    steps = config.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError("config.steps must be a non-empty list")
    unknown = sorted(set(steps).difference(WORKFLOW_ORDER))
    if unknown:
        raise ValueError(f"Unknown workflow steps: {unknown}")
    return config


def _path(config_path: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _input(config: dict[str, Any], config_path: Path, key: str) -> Path:
    path = _path(config_path, config.get("inputs", {}).get(key))
    if path is None:
        raise ValueError(f"Missing inputs.{key}")
    return path


def _output(config: dict[str, Any], config_path: Path, key: str) -> Path:
    path = _path(config_path, config.get("outputs", {}).get(key))
    if path is None:
        raise ValueError(f"Missing outputs.{key}")
    return path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label}: {path}")


def _csv(values: list[Any] | None) -> str | None:
    return None if not values else ",".join(str(value) for value in values)


def _append(arguments: list[str], flag: str, value: Any) -> None:
    if value is not None:
        arguments.extend([flag, str(value)])


def _append_list(arguments: list[str], flag: str, values: list[Any] | None) -> None:
    value = _csv(values)
    if value:
        arguments.extend([flag, value])


def _append_true(arguments: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        arguments.append(flag)


@contextmanager
def _argv(arguments: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = arguments
    try:
        yield
    finally:
        sys.argv = previous


def _run_cli(script: Path, arguments: list[str], dry_run: bool) -> None:
    command = [sys.executable, str(script), *arguments]
    print("[pipeline-stage2]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, env=os.environ.copy())


def validate(config: dict[str, Any], config_path: Path) -> None:
    steps = set(config["steps"])
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    produced = {
        "build_features": "gene_features_dir",
        "prepare_paired_timepoints": "paired_input_root",
        "run_primary_residual_cv": "primary_run_dir",
        "run_paired_6h_residual_cv": "paired_6h_run_dir",
        "run_paired_24h_residual_cv": "paired_24h_run_dir",
        "run_calibration_sensitivity": "calibration_sensitivity_dir",
    }
    for step, key in produced.items():
        if step in steps:
            path = _path(config_path, outputs.get(key))
            if path is None:
                raise ValueError(f"Missing outputs.{key}")
            if path.exists():
                raise FileExistsError(
                    f"Enabled step {step!r} refuses to overwrite existing output: {path}"
                )

    if "build_features" in steps or "prepare_paired_timepoints" in steps:
        for key in ("aggregate_h5ad", "compound_info"):
            _require_file(_input(config, config_path, key), f"inputs.{key}")
    if "prepare_paired_timepoints" in steps:
        _require_file(_input(config, config_path, "tox21_smiles"), "inputs.tox21_smiles")
        _require_file(_input(config, config_path, "stage1_split"), "inputs.stage1_split")

    if "build_features" in steps:
        _require_dir(_input(config, config_path, "cohort_dir"), "inputs.cohort_dir")
    if "run_primary_residual_cv" in steps:
        _require_file(_input(config, config_path, "chem_offset"), "inputs.chem_offset")
        _require_dir(_input(config, config_path, "cohort_dir"), "inputs.cohort_dir")
        if "build_features" not in steps:
            _require_dir(_output(config, config_path, "gene_features_dir"), "outputs.gene_features_dir")

    for step in ("run_paired_6h_residual_cv", "run_paired_24h_residual_cv"):
        if step in steps and "prepare_paired_timepoints" not in steps:
            _require_dir(_output(config, config_path, "paired_input_root"), "outputs.paired_input_root")
        if step in steps:
            _require_file(_input(config, config_path, "chem_offset"), "inputs.chem_offset")

    if "run_calibration_sensitivity" in steps:
        sensitivity = config.get("calibration_sensitivity", {})
        source = _path(config_path, sensitivity.get("source_run_dir"))
        if source is None:
            source = _output(config, config_path, "primary_run_dir")
        if "run_primary_residual_cv" not in steps:
            _require_dir(source, "calibration_sensitivity.source_run_dir")
    
    residual = config.get("residual_learning", {})
    if int(residual.get("outer_folds", 5)) != 5:
        raise ValueError("The bundled publication workflow requires outer_folds=5")
    seeds = residual.get("outer_seeds", [])
    if not seeds:
        raise ValueError("residual_learning.outer_seeds must not be empty")
    print("[pipeline-stage2] configuration valid")


def _build_features(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    output = _output(config, config_path, "gene_features_dir")
    print(f"[pipeline-stage2] build_features -> {output}")
    if dry_run:
        return
    module = build_gene_features
    settings = config.get("feature_construction", {})
    module.AGGREGATE_H5AD = str(_input(config, config_path, "aggregate_h5ad"))
    module.COMPOUNDINFO_PATH = str(_input(config, config_path, "compound_info"))
    module.CELL_IDS = list(settings.get("cells", module.CELL_IDS))
    module.PERT_TIME = float(settings.get("exposure_time_hours", module.PERT_TIME))
    module.DOSE_WINDOW = tuple(float(value) for value in settings.get("dose_window_um", module.DOSE_WINDOW))
    module.REPLICATE_AGG_METHOD = str(settings.get("replicate_aggregation", module.REPLICATE_AGG_METHOD))
    module.main(_input(config, config_path, "cohort_dir"), output)


def _prepare_paired(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    output = _output(config, config_path, "paired_input_root")
    print(f"[pipeline-stage2] prepare_paired_timepoints -> {output}")
    if dry_run:
        return
    module = prepare_paired_timepoint_inputs
    settings = config.get("paired_timepoints", {})
    module.COMPOUNDINFO_PATH = _input(config, config_path, "compound_info")
    module.TOX21_SMILES_PATH = _input(config, config_path, "tox21_smiles")
    module.CELL_IDS = tuple(settings.get("cells", module.CELL_IDS))
    module.TIMEPOINTS_HOURS = tuple(float(value) for value in settings.get("times_hours", module.TIMEPOINTS_HOURS))
    module.DOSE_WINDOW_UM = tuple(float(value) for value in settings.get("dose_window_um", module.DOSE_WINDOW_UM))
    with _argv(
        [
            "prepare_paired_timepoint_inputs.py",
            "--adata-path",
            str(_input(config, config_path, "aggregate_h5ad")),
            "--stage1-split-path",
            str(_input(config, config_path, "stage1_split")),
            "--output-root",
            str(output),
        ]
    ):
        module.main()


def _residual_arguments(
    config: dict[str, Any], config_path: Path, *, out_root: Path, cohort_dir: Path,
    feature_dir: Path, paired_time: int | None = None,
) -> list[str]:
    settings = config.get("residual_learning", {})
    arguments: list[str] = []
    if paired_time is not None:
        arguments.extend(
            [
                "--pert-time",
                str(paired_time),
                "--input-root",
                str(_output(config, config_path, "paired_input_root")),
            ]
        )
    arguments.extend(
        [
            "--out-root",
            str(out_root),
            "--chem-offset-path",
            str(_input(config, config_path, "chem_offset")),
        ]
    )
    if paired_time is None:
        arguments.extend(
            [
                "--cohort-dir",
                str(cohort_dir),
                "--gene-features-dir",
                str(feature_dir),
            ]
        )
        arguments.extend(["--ridge-solver", str(settings.get("ridge_solver", "lbfgs"))])
    # CEViChE-residualized variant is deprecated; unadjusted ("standardized") is
    # now the only supported mode. Original default kept commented for provenance.
    # arguments.extend(["--variant", str(settings.get("variant", "both"))])
    arguments.extend(["--variant", str(settings.get("variant", "standardized"))])
    _append_list(arguments, "--tasks", settings.get("tasks"))
    cells = (
        config.get("paired_timepoints", {}).get("cells")
        if paired_time is not None
        else settings.get("cells")
    )
    _append_list(arguments, "--cells", cells)
    _append_list(arguments, "--outer-seeds", settings.get("outer_seeds"))
    _append(arguments, "--seed-search-attempts", settings.get("seed_search_attempts", 1000))
    _append(arguments, "--n-jobs", settings.get("n_jobs", 1))
    _append(arguments, "--points-per-decade", settings.get("points_per_decade", 2.0))
    _append(arguments, "--maxiter", settings.get("maxiter", 500))
    _append(arguments, "--retry-maxiter", settings.get("retry_maxiter", 2000))
    _append(arguments, "--gtol", settings.get("gtol", 1e-6))
    _append(arguments, "--acceptance-gradient", settings.get("acceptance_gradient", 1e-5))
    _append(arguments, "--ftol", settings.get("ftol", 1e-14))
    _append_true(arguments, "--use-sample-weight", bool(settings.get("use_sample_weight", False)))
    _append_true(arguments, "--skip-summary", bool(settings.get("skip_summary", False)))
    return arguments


def _run_primary(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    arguments = _residual_arguments(
        config,
        config_path,
        out_root=_output(config, config_path, "primary_run_dir"),
        cohort_dir=_input(config, config_path, "cohort_dir"),
        feature_dir=_output(config, config_path, "gene_features_dir"),
    )
    _run_cli(
        STAGE2_RUN / "run_repeated_calibrated.py",
        ["primary", *arguments],
        dry_run,
    )


def _run_paired(config: dict[str, Any], config_path: Path, time_hours: int, dry_run: bool) -> None:
    key = "paired_6h_run_dir" if time_hours == 6 else "paired_24h_run_dir"
    arguments = _residual_arguments(
        config,
        config_path,
        out_root=_output(config, config_path, key),
        cohort_dir=Path("."),
        feature_dir=Path("."),
        paired_time=time_hours,
    )
    _run_cli(
        STAGE2_RUN / "run_repeated_calibrated.py",
        ["paired-innerseed", *arguments],
        dry_run,
    )


def _run_calibration_sensitivity(
    config: dict[str, Any], config_path: Path, dry_run: bool
) -> None:
    settings = config.get("calibration_sensitivity", {})
    residual = config.get("residual_learning", {})
    source = _path(config_path, settings.get("source_run_dir"))
    if source is None:
        source = _output(config, config_path, "primary_run_dir")
    arguments = [
        "--out-dir",
        str(_output(config, config_path, "calibration_sensitivity_dir")),
        "--source-run-dir",
        str(source),
        "--chem-offset-path",
        str(_input(config, config_path, "chem_offset")),
        "--cohort-dir",
        str(_input(config, config_path, "cohort_dir")),
        "--gene-features-dir",
        str(_output(config, config_path, "gene_features_dir")),
        "--variant",
        # CEViChE-residualized variant is deprecated; "standardized" is now the
        # only supported mode. Original default kept commented for provenance.
        # str(settings.get("variant", residual.get("variant", "both"))),
        str(settings.get("variant", residual.get("variant", "standardized"))),
        "--expected-repeats",
        str(int(settings.get("expected_repeats", len(residual.get("outer_seeds", []))))),
        "--n-jobs",
        str(int(settings.get("n_jobs", residual.get("n_jobs", 1)))),
        "--points-per-decade",
        str(float(settings.get("points_per_decade", residual.get("points_per_decade", 2.0)))),
        "--maxiter",
        str(int(settings.get("maxiter", residual.get("maxiter", 500)))),
        "--retry-maxiter",
        str(int(settings.get("retry_maxiter", residual.get("retry_maxiter", 2000)))),
        "--gtol",
        str(float(settings.get("gtol", residual.get("gtol", 1e-6)))),
        "--acceptance-gradient",
        str(float(settings.get("acceptance_gradient", residual.get("acceptance_gradient", 1e-5)))),
        "--ftol",
        str(float(settings.get("ftol", residual.get("ftol", 1e-14)))),
    ]
    _append_list(arguments, "--tasks", settings.get("tasks") or residual.get("tasks"))
    _append_list(arguments, "--cells", settings.get("cells") or residual.get("cells"))
    _append(arguments, "--repeat-limit", settings.get("repeat_limit"))
    _run_cli(STAGE2_RUN / "run_stage2_calibration_ridge_ablation.py", arguments, dry_run)


def run(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    selected = set(config["steps"])
    runners = {
        "build_features": lambda: _build_features(config, config_path, dry_run),
        "prepare_paired_timepoints": lambda: _prepare_paired(config, config_path, dry_run),
        "run_primary_residual_cv": lambda: _run_primary(config, config_path, dry_run),
        "run_paired_6h_residual_cv": lambda: _run_paired(config, config_path, 6, dry_run),
        "run_paired_24h_residual_cv": lambda: _run_paired(config, config_path, 24, dry_run),
        "run_calibration_sensitivity": lambda: _run_calibration_sensitivity(config, config_path, dry_run),
    }
    for step in config["steps"]:
        runners[step]()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone transcriptomic residual-learning workflow."
    )
    parser.add_argument("--config", required=True, type=Path)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate inputs and output safety.")
    run_parser = subparsers.add_parser("run", help="Execute configured stages.")
    run_parser.add_argument(
        "--dry-run", action="store_true", help="Print the execution plan only."
    )
    return parser.parse_args()


def main() -> None:
    _prepare_runtime()
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    if args.command == "run":
        if not args.dry_run:
            validate(config, config_path)
        run(config, config_path, args.dry_run)
    else:
        validate(config, config_path)


if __name__ == "__main__":
    main()
