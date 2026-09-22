#!/usr/bin/env python3
"""
wrapper.py
==========
Orchestrate the LINCS preprocessing pipeline:
  load -> embed -> filter -> merge -> select -> align -> aggregate

Each step is run as a subprocess so it gets a clean memory space.
Steps can be skipped via --skip.  Config and run-keys are forwarded
to every child script via --config / --run-keys.

Usage examples
--------------
# full pipeline, all run types
python wrapper.py --config .../lincs_ext.json

# skip embed (faster), chemical only
python wrapper.py --config .../lincs_ext.json --skip embed --run-keys chemical

# only merge + downstream
python wrapper.py --skip load embed filter
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

STEPS = [
    "load",
    "embed",
    "merge",
    "filter",   # filter.py runs both QC-filter AND select
    "align",
    "aggregate",
    "join",
]

SCRIPT_MAP = {
    "load":      SCRIPT_DIR / "load.py",
    "embed":     SCRIPT_DIR / "embed.py",
    "filter":    SCRIPT_DIR / "filter.py",
    "merge":     SCRIPT_DIR / "merge.py",
    "align":     SCRIPT_DIR / "align.py",
    "aggregate": SCRIPT_DIR / "aggregate.py",
    "join":      SCRIPT_DIR / "join_ext.py",
}


def main():
    parser = argparse.ArgumentParser(
        description="LINCS preprocessing pipeline wrapper",
    )
    parser.add_argument(
        "--config",
        default="/home/kyungan/scripts/BioTox/bio/configs/lincs_ext.json",
        help="Path to lincs config JSON",
    )
    parser.add_argument(
        "--run-keys",
        nargs="*",
        default=None,
        help="Run types to process (default: all in config)",
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=STEPS,
        help="Steps to skip (e.g. --skip embed)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        choices=STEPS,
        help="Run ONLY these steps (overrides --skip)",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use (default: same as wrapper)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    run_keys = args.run_keys or list(cfg["runs"].keys())

    if args.only:
        active = [s for s in STEPS if s in args.only]
    else:
        active = [s for s in STEPS if s not in args.skip]

    print(f"Config   : {args.config}")
    print(f"Run keys : {run_keys}")
    print(f"Steps    : {' -> '.join(active)}")
    print(f"Skipped  : {[s for s in STEPS if s not in active] or 'none'}")
    print(f"Python   : {args.python}")
    print(f"{'='*60}\n")

    for step in active:
        script = SCRIPT_MAP[step]
        if not script.exists():
            print(f"[ERROR] script not found: {script}")
            sys.exit(1)

        cmd = [
            args.python,
            str(script),
            "--config", args.config,
        ]
        if args.run_keys:
            cmd += ["--run-keys"] + args.run_keys

        print(f"{'='*60}")
        print(f"  STEP: {step}")
        print(f"  CMD : {' '.join(cmd)}")
        print(f"{'='*60}\n")

        t0 = time.time()
        result = subprocess.run(cmd)
        elapsed = time.time() - t0

        if result.returncode != 0:
            print(f"\n[FAILED] {step} exited with code {result.returncode} "
                  f"after {elapsed:.0f}s")
            sys.exit(result.returncode)

        print(f"\n[OK] {step} finished in {elapsed:.0f}s\n")

    print(f"{'='*60}")
    print("Pipeline complete.")


if __name__ == "__main__":
    main()
