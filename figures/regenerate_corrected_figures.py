#!/usr/bin/env python3
"""Regenerate Figures 2-5 and Figure S2 from their own canonical source data.

Each figure already has a self-contained ``generate_*.py`` script under
``FigureN/code/`` that reads from ``FigureN/source_data/`` (or another
checked-in path) and renders that figure. This wrapper's only job is to
invoke each requested script with the right arguments and, where a script
writes under a working ``generation/`` subdirectory with its own internal
file names, copy the results to the canonical ``FigureN.pdf`` / ``.png`` /
``.svg`` names. Each generator's source-data paths are read from that
figure's ``code/config.json``; ``figures_config.json`` contains only wrapper
routing and output-copy rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

FIGURES_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = FIGURES_ROOT / "figures_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_figure_config(name: str, spec: dict) -> Tuple[dict, Dict[str, object]]:
    """Load a figure's canonical config and resolve every configured input."""
    config_path = FIGURES_ROOT / spec["config"]
    if not config_path.is_file():
        raise FileNotFoundError(f"{name}: figure config not found: {config_path}")
    figure_config = json.loads(config_path.read_text(encoding="utf-8"))
    code_dir = config_path.parent
    source_paths = {}
    for key, value in figure_config.get("input", {}).items():
        if not isinstance(value, str):
            raise TypeError(f"{name}: input config value {key!r} must be a path string")
        path = (code_dir / value).resolve()
        if not path.exists():
            raise FileNotFoundError(f"{name}: configured input does not exist: {path}")
        if path.is_dir():
            files = sorted(item for item in path.iterdir() if item.is_file())
            if not files:
                raise FileNotFoundError(f"{name}: configured input directory is empty: {path}")
            source_paths[key] = [str(item) for item in files]
        else:
            source_paths[key] = str(path)
    configured_files = figure_config.get("input_files", {})
    if not isinstance(configured_files, dict):
        raise TypeError(f"{name}: input_files must be an object")
    input_inventory = {}
    for key, values in configured_files.items():
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise TypeError(f"{name}: input_files[{key!r}] must be a list of path strings")
        resolved = []
        for value in values:
            path = (code_dir / value).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"{name}: configured input file does not exist: {path}")
            resolved.append(str(path))
        input_inventory[key] = resolved
    source_paths["_configured_files"] = input_inventory
    return figure_config, source_paths


def run_figure(name: str, spec: dict, python: str) -> dict:
    script = FIGURES_ROOT / spec["script"]
    if not script.is_file():
        raise FileNotFoundError(f"{name}: generator script not found: {script}")
    figure_config, source_paths = load_figure_config(name, spec)

    run_dir = FIGURES_ROOT / spec["run_dir"] if spec.get("run_dir") else None
    if run_dir is not None and spec.get("clear_run_dir_first"):
        shutil.rmtree(run_dir, ignore_errors=True)

    args = [str(script)]
    for arg in spec.get("args", []):
        args.append(arg.replace("{run_dir}", str(run_dir)) if run_dir else arg)

    print(f"[{name}] source data:", flush=True)
    for key, path in source_paths.items():
        if isinstance(path, list):
            print(f"  {key}:", flush=True)
            for item in path:
                print(f"    {item}", flush=True)
        else:
            print(f"  {key}: {path}", flush=True)
    print(f"[{name}] running: {python} {' '.join(args)}", flush=True)
    subprocess.run([python, *args], check=True)

    copied = {}
    for rule in spec.get("copy", []):
        source = (run_dir if run_dir else FIGURES_ROOT / name) / rule["from"]
        destination = FIGURES_ROOT / name / rule["to"]
        if not source.is_file():
            raise FileNotFoundError(f"{name}: expected output missing: {source}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[rule["to"]] = str(destination)

    outputs = {}
    figure_dir = FIGURES_ROOT / name
    for suffix in (".pdf", ".png", ".svg"):
        candidate = figure_dir / f"{name}{suffix}"
        if candidate.is_file():
            outputs[candidate.name] = {"bytes": candidate.stat().st_size, "sha256": sha256(candidate)}
    if not outputs:
        raise FileNotFoundError(f"{name}: no Figure*.{{pdf,png,svg}} present in {figure_dir} after the run")
    configured_outputs = figure_config.get("output", {}).get("files", [])
    if not isinstance(configured_outputs, list) or not all(isinstance(value, str) for value in configured_outputs):
        raise TypeError(f"{name}: output.files must be a list of relative paths")
    output_root = (figure_config.get("output", {}).get("directory") or ".")
    output_dir = ((FIGURES_ROOT / spec["config"]).parent / output_root).resolve()
    for value in configured_outputs:
        path = (output_dir / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{name}: configured output file does not exist: {path}")
    return {
        "config": str((FIGURES_ROOT / spec["config"]).resolve()),
        "inputs": source_paths,
        "output_config": figure_config.get("output", {}),
        "copied": copied,
        "outputs": outputs,
    }


def main() -> None:
    config = load_config()
    available = list(config["figures"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures",
        default=",".join(available),
        help=f"Comma-separated subset of {{{','.join(available)}}} (default: all)",
    )
    args = parser.parse_args()
    requested = [item.strip() for item in args.figures.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise SystemExit(f"Unknown figure(s) {unknown}; choose from {available}")

    python = config["python"]
    report = {}
    for name in requested:
        report[name] = run_figure(name, config["figures"][name], python)
        print(f"[{name}] done: {list(report[name]['outputs'])}", flush=True)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"Figure generation failed: {error}")
