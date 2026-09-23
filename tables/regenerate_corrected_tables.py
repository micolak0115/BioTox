#!/usr/bin/env python3
"""Regenerate the active publication tables from their table-local configs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


TABLES_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = TABLES_ROOT / "tables_config.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def validate_table_config(name: str, spec: dict) -> tuple[Path, dict]:
    config_path = TABLES_ROOT / spec["config"]
    if not config_path.is_file():
        raise FileNotFoundError(f"{name}: missing table config: {config_path}")
    table_config = json.loads(config_path.read_text(encoding="utf-8"))
    input_files = table_config.get("input_files", {})
    if not isinstance(input_files, dict):
        raise TypeError(f"{name}: input_files must be an object")
    for key, value in input_files.items():
        values = value if isinstance(value, list) else [value]
        if not values or not all(isinstance(item, str) for item in values):
            raise TypeError(f"{name}: input_files[{key!r}] must contain path strings")
        for item in values:
            path = resolve_config_path(config_path, item)
            if not path.exists():
                raise FileNotFoundError(f"{name}: missing configured input {key}: {path}")
            if path.is_dir() and not any(path.iterdir()):
                raise FileNotFoundError(f"{name}: configured input directory is empty: {path}")
    return config_path, table_config


def run_table(name: str, spec: dict, python: str) -> dict:
    script = TABLES_ROOT / spec["script"]
    if not script.is_file():
        raise FileNotFoundError(f"{name}: missing generator: {script}")
    config_path, table_config = validate_table_config(name, spec)
    run_dir = TABLES_ROOT / spec["run_dir"]
    if spec.get("clear_run_dir_first"):
        shutil.rmtree(run_dir, ignore_errors=True)
    args = [
        str(script),
        *[
            argument.replace("{run_dir}", str(run_dir))
            for argument in spec.get("args", [])
        ],
    ]
    print(f"[{name}] config: {config_path}", flush=True)
    print(f"[{name}] running: {python} {' '.join(args)}", flush=True)
    subprocess.run([python, *args], check=True)

    outputs = {}
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            outputs[str(path.relative_to(run_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
    if not outputs:
        raise FileNotFoundError(f"{name}: generator produced no files in {run_dir}")
    return {
        "config": str(config_path),
        "generator": str(script),
        "run_dir": str(run_dir),
        "inputs": table_config.get("input_files", {}),
        "outputs": outputs,
    }


def main() -> None:
    config = load_config()
    available = list(config["tables"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tables",
        default=",".join(available),
        help=f"Comma-separated subset of {{{','.join(available)}}}; Table1 is static.",
    )
    args = parser.parse_args()
    requested = [item.strip() for item in args.tables.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available) - set(config.get("static_tables", [])))
    if unknown:
        raise SystemExit(f"Unknown table(s): {unknown}")
    if any(item in config.get("static_tables", []) for item in requested):
        raise SystemExit("Table1 is static manuscript LaTeX and has no generator.")

    report = {}
    for name in requested:
        report[name] = run_table(name, config["tables"][name], config["python"])
        print(f"[{name}] done: {len(report[name]['outputs'])} files", flush=True)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"Table generation failed: {error}")
