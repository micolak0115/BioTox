#!/usr/bin/env python3
"""Unified command-line entry point for calibrated BioTox Stage-2 runs.

The implementation modules remain separate because they are also imported by
tests, calibration-sensitivity analysis, and post-hoc reporting. This command
provides one pipeline-facing entry point while preserving each existing
runner's complete CLI and behavior through lazy delegation.

Examples
--------
Primary 20-repeat analysis::

    python pipeline/stage2/run_repeated_calibrated.py primary --out-root ...

Paired 6-hour/24-hour analysis with the historical inner-seed recovery::

    python pipeline/stage2/run_repeated_calibrated.py paired-innerseed \
        --pert-time 6 --input-root ... --out-root ...

The legacy runner scripts remain valid compatibility entry points.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))


MODES: dict[str, tuple[str, str]] = {
    "primary": (
        "stage2.run_repeated_calibrated_pipeline",
        "primary 20-repeat, 5-fold calibrated nested CV",
    ),
    "paired": (
        "stage2.run_repeated_calibrated_timepoint",
        "paired 6-hour/24-hour calibrated nested CV",
    ),
    "paired-innerseed": (
        "stage2.run_repeated_calibrated_timepoint_innerseed",
        "paired CV with deterministic inner-seed recovery",
    ),
    "outer-fold": (
        "stage2.nested_cv_offset_logistic_calibrated",
        "single non-repeated calibrated outer-fold analysis",
    ),
    "prepare-paired-inputs": (
        "stage2.prepare_paired_timepoint_inputs",
        "leakage-safe paired 6-hour/24-hour input preparation",
    ),
}


def _load_main(module_name: str) -> Callable[[], None]:
    module = __import__(module_name, fromlist=["main"])
    try:
        return module.main
    except AttributeError as exc:  # pragma: no cover - defensive contract check
        raise RuntimeError(f"Unified mode target has no main(): {module_name}") from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry point for BioTox calibrated Stage-2 runs. "
            "All mode-specific options are forwarded unchanged to the "
            "original runner."
        )
    )
    parser.add_argument("mode", choices=sorted(MODES), help="Run mode")
    return parser


def main() -> None:
    # Parse only the mode here. The delegated runner owns its original parser,
    # defaults, validation, and options; this avoids duplicating contracts and
    # prevents the unified wrapper from silently dropping a runner feature.
    if len(sys.argv) > 1 and sys.argv[1] in MODES:
        mode = sys.argv[1]
        delegated_argv = sys.argv[2:]
    else:
        args, delegated_argv = _parser().parse_known_args()
        mode = args.mode
    module_name, _ = MODES[mode]
    sys.argv = [f"{module_name}.py", *delegated_argv]
    _load_main(module_name)()


if __name__ == "__main__":
    main()
