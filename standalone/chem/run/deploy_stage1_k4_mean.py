"""Deploy the frozen task-specific K=4 Stage-1 chemical ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


HERE = Path(__file__).resolve().parent
_STANDALONE_ROOT = HERE.parents[1]


def _register_bundled_package(name: str, *locations: Path) -> None:
    """Make a bundled sibling ``run/`` directory importable under its
    original package name.

    An in-memory ``sys.modules`` entry alone is invisible to spawned
    worker processes (e.g. joblib/loky candidate-fitting workers):
    each starts a fresh interpreter that must resolve
    ``publication.X`` / ``chem.X`` imports itself. This also creates
    ONE namespace-package directory
    (``_STANDALONE_ROOT/_pkg_alias/<name>``, no ``__init__.py`` of
    its own) containing a flat, relative on-disk symlink to every
    ``.py`` file from every ``location``, and adds its PARENT
    directory to ``sys.path`` and ``PYTHONPATH`` -- ``sys.path``
    because multiprocessing's spawn start method propagates the
    parent's ``sys.path`` to workers rather than re-deriving it from
    ``PYTHONPATH``; ``PYTHONPATH`` for plain ``subprocess.run`` stage
    launches, which DO read it at their own interpreter startup.
    """
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(location) for location in locations]
        sys.modules[name] = module

    alias_root = _STANDALONE_ROOT / "_pkg_alias"
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


_register_bundled_package("chem", HERE)
_register_bundled_package("publication", _STANDALONE_ROOT / "bio" / "run", HERE)

from publication.chem_stage1_per_task_offset import (  # noqa: E402
    COHORT_DIR,
    load_cohort_union,
)


DEFAULT_STAGE1_DIR = (
    HERE / "_run_output" / "chem_stage1_candidates_fresh_v1"
)
DEFAULT_REFERENCE = (
    HERE
    / "_run_output"
    / "stage1_topk_summary_fresh_v1"
    / "stage1_topk_k4_reference_by_task.csv"
)
DEFAULT_OUTPUT_DIR = HERE / "_run_output" / "chem_offset_k4_deployed_fresh_v1"
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _to_logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def _load_cache(path: Path) -> dict:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    required = {
        "candidate_names",
        "test_preds_by_candidate",
        "cohort_preds_by_candidate",
        "cohort_ik",
        "y_test",
        "seed",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"{path} is missing cache keys: {sorted(missing)}")
    if int(cache["seed"]) != 1:
        raise ValueError(f"{path} was not generated with fixed seed 1")
    return cache


def deploy(
    stage1_dir: Path,
    reference_path: Path,
    output_dir: Path,
    cohort_dir: Path,
    *,
    k: int = 4,
) -> None:
    stage1_dir = stage1_dir.resolve()
    reference_path = reference_path.resolve()
    output_dir = output_dir.resolve()
    cohort_dir = cohort_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_dir}")
    if not reference_path.is_file():
        raise FileNotFoundError(reference_path)

    reference = pd.read_csv(reference_path)
    required_columns = {"task", "ranked_candidates"}
    missing_columns = required_columns.difference(reference.columns)
    if missing_columns:
        raise ValueError(
            f"K=4 reference is missing columns: {sorted(missing_columns)}"
        )
    if set(reference["task"]) != set(TASKS):
        raise ValueError("K=4 reference does not contain exactly 12 tasks")

    cohort_ids = load_cohort_union(cohort_dir)["ik"].astype(str).to_numpy()
    offset_frame: pd.DataFrame | None = None
    deployment_rows = []
    cache_checksums = {}
    for task in TASKS:
        cache_path = stage1_dir / "task_benchmark_cache" / f"{task}.pkl"
        cache = _load_cache(cache_path)
        cache_checksums[task] = _sha256(cache_path)
        row = reference.loc[reference["task"] == task].iloc[0]
        ranked = [
            value.strip()
            for value in str(row["ranked_candidates"]).split(";")
            if value.strip()
        ]
        selected = ranked[:k]
        if len(selected) != k:
            raise ValueError(f"{task}: expected {k} selected candidates")
        available = set(cache["candidate_names"])
        missing = sorted(set(selected) - available)
        if missing:
            raise ValueError(f"{task}: missing cached candidates {missing}")

        test_probability = np.mean(
            [cache["test_preds_by_candidate"][name] for name in selected],
            axis=0,
        )
        cohort_probability = np.mean(
            [cache["cohort_preds_by_candidate"][name] for name in selected],
            axis=0,
        )
        task_frame = pd.DataFrame(
            {
                "ik": np.asarray(cache["cohort_ik"]).astype(str),
                f"{task}_chem_logit": _to_logit(cohort_probability),
            }
        )
        if len(task_frame) != len(cohort_ids):
            raise ValueError(f"{task}: cohort ID length mismatch")
        if set(task_frame["ik"]) != set(cohort_ids):
            raise ValueError(f"{task}: cached cohort identifier set mismatch")

        if offset_frame is None:
            offset_frame = task_frame
        else:
            if not np.array_equal(
                offset_frame["ik"].astype(str).to_numpy(),
                task_frame["ik"].astype(str).to_numpy(),
            ):
                raise ValueError(f"{task}: cohort identifiers are misaligned")
            offset_frame[f"{task}_chem_logit"] = task_frame[
                f"{task}_chem_logit"
            ].to_numpy()

        deployment_rows.append(
            {
                "task": task,
                "rule": "mean",
                "k": k,
                "ensemble_candidates": ";".join(selected),
                "test_auprc": float(
                    average_precision_score(cache["y_test"], test_probability)
                ),
            }
        )

    if offset_frame is None:
        raise RuntimeError("No Stage-1 tasks were deployed")
    output_dir.mkdir(parents=True, exist_ok=False)
    offset_path = output_dir / "chem_offset_final.csv"
    report_path = output_dir / "chem_prior_deployment_report.csv"
    offset_frame.to_csv(offset_path, index=False)
    pd.DataFrame(deployment_rows).to_csv(report_path, index=False)
    manifest = {
        "pipeline": "stage1_k4_equal_weight_deployment",
        "fixed_model_seed": 1,
        "ensemble_rule": "arithmetic_mean",
        "k": k,
        "n_tasks": len(TASKS),
        "stage1_dir": str(stage1_dir),
        "cohort_dir": str(cohort_dir),
        "reference_path": str(reference_path),
        "reference_sha256": _sha256(reference_path),
        "cache_sha256_by_task": cache_checksums,
        "offset_sha256": _sha256(offset_path),
        "deployment_report_sha256": _sha256(report_path),
        "analysis_complete": True,
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the frozen per-task K=4 equal-weight chemical offset from "
            "the fixed-seed Stage-1 candidate caches."
        )
    )
    parser.add_argument("--stage1-dir", type=Path, default=DEFAULT_STAGE1_DIR)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=COHORT_DIR,
        help="Directory produced by build_matched_cohort.py.",
    )
    parser.add_argument("--k", type=int, default=4)
    args = parser.parse_args()
    deploy(
        args.stage1_dir,
        args.reference,
        args.output_dir,
        args.cohort_dir,
        k=args.k,
    )


if __name__ == "__main__":
    main()
