#!/usr/bin/env python3

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
STAGE1_RUN = ROOT / "stage1"
STAGE2_RUN = ROOT / "stage2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage1 import build_matched_cohort
from stage1 import chem_stage1_per_task_offset

WORKFLOW_ORDER = (
    "build_cohort",
    "fit_candidates",
    "select_ensemble",
    "deploy_ensemble",
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


def _require_file(path: Path | None, label: str) -> None:
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{label}: {path}")


def _require_dir(path: Path | None, label: str) -> None:
    if path is None or not path.is_dir():
        raise FileNotFoundError(f"{label}: {path}")


def _output_path(config: dict[str, Any], config_path: Path, key: str) -> Path:
    value = config.get("outputs", {}).get(key)
    path = _path(config_path, value)
    if path is None:
        raise ValueError(f"Missing outputs.{key}")
    return path


def _input_path(config: dict[str, Any], config_path: Path, key: str) -> Path:
    value = config.get("inputs", {}).get(key)
    path = _path(config_path, value)
    if path is None:
        raise ValueError(f"Missing inputs.{key}")
    return path


def _model_reference(config_path: Path, value: str) -> str:
    """Resolve local model directories while preserving Hub model identifiers."""
    expanded = os.path.expandvars(os.path.expanduser(value))
    candidate = Path(expanded)
    if candidate.is_absolute() or expanded.startswith((".", "~")):
        return str(_path(config_path, expanded))
    relative = config_path.parent / candidate
    if relative.exists():
        return str(relative.resolve())
    return value


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
    print("[pipeline-stage1]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True, env=os.environ.copy())


def validate(config: dict[str, Any], config_path: Path) -> None:
    steps = set(config["steps"])
    inputs = config.get("inputs", {})
    outputs = config.get("outputs", {})
    for key in ("aggregate_h5ad", "compound_info", "tox21_smiles"):
        if "build_cohort" in steps:
            _require_file(_path(config_path, inputs.get(key)), f"inputs.{key}")

    if "fit_candidates" in steps:
        gnn_root = _path(config_path, inputs.get("pretrained_gnn_root"))
        _require_dir(gnn_root, "inputs.pretrained_gnn_root")
        _require_file(
            _path(config_path, inputs.get("chemprop_python")),
            "inputs.chemprop_python",
        )
        chemberta_model = inputs.get("chemberta_model")
        if not isinstance(chemberta_model, str) or not chemberta_model:
            raise ValueError("Missing inputs.chemberta_model")
        model_reference = _model_reference(config_path, chemberta_model)
        if Path(model_reference).is_absolute():
            _require_dir(Path(model_reference), "inputs.chemberta_model")
        cohort = _path(config_path, outputs.get("cohort_dir"))
        if "build_cohort" not in steps:
            _require_dir(cohort, "outputs.cohort_dir")

    dependencies = {
        "select_ensemble": ("fit_candidates", "stage1_dir"),
        "deploy_ensemble": ("select_ensemble", "ensemble_selection_dir"),
    }
    for step, (producer, output_key) in dependencies.items():
        if step in steps and producer not in steps:
            _require_dir(
                _path(config_path, outputs.get(output_key)),
                f"outputs.{output_key}",
            )

    if "deploy_ensemble" in steps and "fit_candidates" not in steps:
        _require_dir(
            _path(config_path, outputs.get("stage1_dir")),
            "outputs.stage1_dir",
        )

    for step, key in zip(
        WORKFLOW_ORDER,
        ("cohort_dir", "stage1_dir", "ensemble_selection_dir", "deployment_dir"),
    ):
        if step in steps:
            output = _path(config_path, outputs.get(key))
            if output is None:
                raise ValueError(f"Missing outputs.{key}")
            if output.exists():
                raise FileExistsError(
                    f"Enabled step {step!r} refuses to overwrite existing output: {output}"
                )

    ensemble = config.get("ensemble", {})
    if int(ensemble.get("k", 4)) != 4:
        raise ValueError("This publication workflow requires ensemble.k=4")
    print("[pipeline-stage1] configuration valid")


def _build_cohort(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    output = _output_path(config, config_path, "cohort_dir")
    print(f"[pipeline-stage1] build_cohort -> {output}")
    if dry_run:
        return
    module = build_matched_cohort
    inputs = config["inputs"]
    settings = config.get("cohort", {})
    module.AGGREGATE_H5AD = str(_path(config_path, inputs["aggregate_h5ad"]))
    module.COMPOUNDINFO_PATH = str(_path(config_path, inputs["compound_info"]))
    module.TOX21_SMILES_PATH = str(_path(config_path, inputs["tox21_smiles"]))
    module.CELL_IDS = list(settings.get("cells", module.CELL_IDS))
    module.PERT_TIME = float(settings.get("exposure_time_hours", module.PERT_TIME))
    module.DOSE_WINDOW = tuple(float(v) for v in settings.get("dose_window_um", module.DOSE_WINDOW))
    module.COHORT_FRAC_TRAIN = float(settings.get("cohort_train_fraction", module.COHORT_FRAC_TRAIN))
    module.COHORT_FRAC_VALID = float(settings.get("cohort_valid_fraction", module.COHORT_FRAC_VALID))
    module.FINETUNE_FRAC_TRAIN = float(settings.get("development_train_fraction", module.FINETUNE_FRAC_TRAIN))
    module.FINETUNE_FRAC_VALID = float(settings.get("development_valid_fraction", module.FINETUNE_FRAC_VALID))
    module.FINETUNE_FRAC_TEST = float(settings.get("development_test_fraction", module.FINETUNE_FRAC_TEST))
    module.main(output)


def _fit_candidates(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    output = _output_path(config, config_path, "stage1_dir")
    cohort = _output_path(config, config_path, "cohort_dir")
    settings = config.get("candidate_training", {})
    print(f"[pipeline-stage1] fit_candidates -> {output}")
    if dry_run:
        return
    inputs = config["inputs"]
    os.environ["BIOTOX_PRETRAIN_GNN_ROOT"] = str(
        _path(config_path, inputs["pretrained_gnn_root"])
    )
    os.environ["BIOTOX_GNN_PRETRAINED_VARIANT"] = str(
        inputs.get("gnn_pretrained_variant", "supervised_contextpred")
    )
    os.environ["BIOTOX_CHEMBERTA_MODEL"] = str(
        _model_reference(
            config_path,
            inputs.get("chemberta_model", "DeepChem/ChemBERTa-100M-MLM"),
        )
    )
    os.environ["BIOTOX_CHEMPROP_PYTHON"] = str(
        _path(config_path, inputs["chemprop_python"])
    )
    os.environ["BIOTOX_CHEMPROP_SCRIPT"] = str(
        STAGE1_RUN / "chemprop_stage1_candidate.py"
    )
    module = chem_stage1_per_task_offset
    tasks = settings.get("tasks") or None
    gpu_ids = settings.get("gpu_ids")
    cache_source = _path(config_path, settings.get("cache_source_dir"))
    module.configure_model_settings(
        settings.get("candidates"), settings.get("model_settings", {})
    )
    module.main(
        tasks=tasks,
        seed=int(settings.get("seed", 1)),
        gpu_ids=gpu_ids,
        candidate_n_jobs=int(settings.get("n_jobs", 1)),
        out_dir=output,
        cohort_dir=cohort,
        cache_source_dir=cache_source,
    )


def _select_ensemble(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    settings = config.get("ensemble", {})
    _run_cli(
        STAGE1_RUN / "plot_topk_ensemble_publication.py",
        [
            "--stage1-dir",
            str(_output_path(config, config_path, "stage1_dir")),
            "--output-dir",
            str(_output_path(config, config_path, "ensemble_selection_dir")),
            "--n-bootstrap",
            str(int(settings.get("n_bootstrap", 2000))),
            "--dpi",
            str(int(settings.get("dpi", 300))),
        ],
        dry_run,
    )


def _deploy_ensemble(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    selection_dir = _output_path(config, config_path, "ensemble_selection_dir")
    _run_cli(
        STAGE1_RUN / "deploy_stage1_k4_mean.py",
        [
            "--stage1-dir",
            str(_output_path(config, config_path, "stage1_dir")),
            "--reference",
            str(selection_dir / "stage1_topk_k4_reference_by_task.csv"),
            "--output-dir",
            str(_output_path(config, config_path, "deployment_dir")),
            "--cohort-dir",
            str(_output_path(config, config_path, "cohort_dir")),
            "--k",
            str(int(config.get("ensemble", {}).get("k", 4))),
        ],
        dry_run,
    )


def run(config: dict[str, Any], config_path: Path, dry_run: bool) -> None:
    selected = set(config["steps"])
    runners = {
        "build_cohort": _build_cohort,
        "fit_candidates": _fit_candidates,
        "select_ensemble": _select_ensemble,
        "deploy_ensemble": _deploy_ensemble,
    }
    for step in config["steps"]:
        runners[step](config, config_path, dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone molecular prediction and K=4 ensemble workflow."
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
