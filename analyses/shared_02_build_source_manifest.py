#!/usr/bin/env python3
"""Write deterministic SHA-256 checksums for the public analysis archive."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).absolute().parent
INDEX = ROOT / "METHODS_ANALYSIS_SCRIPT_INDEX.csv"
MANIFEST = ROOT / "SOURCE_MANIFEST.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with INDEX.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    names = [record["archive_filename"] for record in records]
    names.extend(
        [
            "METHODS_ANALYSIS_SCRIPT_INDEX.csv",
            "README.md",
            "shared_01_check_analysis_integrity.py",
            "shared_02_build_source_manifest.py",
        ]
    )
    lines = []
    for name in names:
        path = ROOT / name
        if not path.is_file():
            raise SystemExit(f"Missing archive file: {path}")
        lines.append(f"{sha256(path)}  {name}")
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {MANIFEST}")


if __name__ == "__main__":
    main()
