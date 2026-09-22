#!/usr/bin/env python3
"""Regenerate the publication/tables bundle from its own canonical inputs.

Mirrors publication/figures/regenerate_corrected_figures.py: a thin,
config-driven wrapper. Script paths and CLI args live in ``tables_config.json``
next to this file; each generator's own input data paths live in
``table_paths.json`` (read by the generators themselves via
``configure_paths()``). Existing publication table assets under ``Table1/``,
``Table2/``, ``TableS1``-``TableS9`` are never touched -- every invocation
writes into a fresh ``--output-dir`` that must not already exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
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


def run_table(name: str, spec: dict, python: str, run_dir: Path) -> dict:
    script = TABLES_ROOT / spec["script"]
    if not script.is_file():
        raise FileNotFoundError(f"{name}: generator script not found: {script}")

    args = [str(script)]
    for arg in spec.get("args", []):
        args.append(arg.replace("{run_dir}", str(run_dir)))

    print(f"[{name}] running: {python} {' '.join(args)}", flush=True)
    subprocess.run([python, *args], check=True)

    outputs = {
        str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }
    if not outputs:
        raise FileNotFoundError(f"{name}: no output files present in {run_dir} after the run")
    return {"run_dir": str(run_dir), "outputs": outputs}


def main() -> None:
    config = load_config()
    available = list(config["tables"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--tables",
        default=",".join(available),
        help=f"Comma-separated subset of {{{','.join(available)}}} (default: all)",
    )
    args = parser.parse_args()
    requested = [item.strip() for item in args.tables.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise SystemExit(f"Unknown table set(s) {unknown}; choose from {available}")
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    python = config["python"]
    report = {}
    for name in requested:
        run_dir = args.output_dir / name
        report[name] = run_table(name, config["tables"][name], python, run_dir)
        print(f"[{name}] done: {len(report[name]['outputs'])} files", flush=True)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG_PATH),
        "table_paths": str(TABLES_ROOT / "table_paths.json"),
        "tables": report,
    }
    (args.output_dir / "REGENERATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({name: list(result["outputs"]) for name, result in report.items()}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(f"Table generation failed: {error}")
