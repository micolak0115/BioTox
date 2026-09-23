#!/usr/bin/env python3
"""Validate the public methods-script archive and its provenance index."""

from __future__ import annotations

import ast
import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).absolute().parent
REPOSITORY = ROOT.parents[1]
INDEX = ROOT / "METHODS_ANALYSIS_SCRIPT_INDEX.csv"
REPORT_CSV = ROOT / "SCRIPT_INTEGRITY_REPORT.csv"
REPORT_MD = ROOT / "SCRIPT_INTEGRITY_REPORT.md"
REQUIRED_COLUMNS = {
    "methods_section",
    "sequence",
    "archive_filename",
    "role",
    "historical_source",
    "sha256",
    "purpose",
}
NAME_PATTERN = re.compile(r"m(?P<minor>\d+)_(?P<sequence>\d{2})_[a-z0-9_]+\.py")
SECTION_DIRECTORY = {
    "2.2": "data_preparation",
    "2.3": "model_specification",
    "2.4": "posthoc_analysis",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with INDEX.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Index is missing columns: {sorted(missing)}")
        records = list(reader)

    seen_names: set[str] = set()
    seen_positions: set[tuple[str, str]] = set()
    report: list[dict[str, str]] = []
    for record in records:
        name = record["archive_filename"]
        script = ROOT / name
        source = REPOSITORY / record["historical_source"]
        errors: list[str] = []
        if name in seen_names:
            errors.append("duplicate filename")
        seen_names.add(name)
        position = (record["methods_section"], record["sequence"])
        if position in seen_positions:
            errors.append("duplicate section/sequence")
        seen_positions.add(position)
        match = NAME_PATTERN.fullmatch(Path(name).name)
        expected_minor = record["methods_section"].split(".")[-1]
        if not match or match.group("minor") != expected_minor:
            errors.append("filename does not match methods section")
        if not match or match.group("sequence") != record["sequence"]:
            errors.append("filename does not match sequence")
        expected_directory = SECTION_DIRECTORY.get(record["methods_section"])
        if expected_directory is None or Path(name).parent.as_posix() != expected_directory:
            errors.append("filename is not stored under its methods-section directory")
        if not script.is_file():
            errors.append("script missing")
            actual_hash = ""
        else:
            actual_hash = sha256(script)
            if actual_hash != record["sha256"]:
                errors.append("script checksum differs from index")
            try:
                text = script.read_text(encoding="utf-8")
                ast.parse(text, filename=name)
                compile(text, name, "exec")
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"syntax/encoding failure: {exc}")
        source_status = "not bundled"
        if source.is_file():
            source_status = "hash match" if sha256(source) == actual_hash else "hash differs"
            # A public snapshot may include documented portability or bug fixes
            # relative to the preserved working copy; retain that distinction
            # as provenance instead of rejecting the release.
            if source_status == "hash differs":
                source_status = "hash differs (historical)"
        report.append(
            {
                "methods_section": record["methods_section"],
                "sequence": record["sequence"],
                "archive_filename": name,
                "role": record["role"],
                "sha256": actual_hash,
                "historical_source": record["historical_source"],
                "source_status": source_status,
                "status": "PASS" if not errors else "FAIL",
                "details": "; ".join(errors) if errors else "name, checksum, and static compile passed",
            }
        )

    indexed = set(seen_names)
    present = {
        path.relative_to(ROOT).as_posix()
        for directory in SECTION_DIRECTORY.values()
        for path in (ROOT / directory).glob("m*.py")
    }
    unindexed = sorted(present - indexed)
    missing_scripts = sorted(indexed - present)
    failures = [row for row in report if row["status"] == "FAIL"]
    if unindexed or missing_scripts:
        failures.append({"status": "FAIL"})

    with REPORT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)

    lines = [
        "# Analysis script integrity report",
        "",
        f"- Indexed scripts: {len(report)}",
        f"- Passed: {sum(row['status'] == 'PASS' for row in report)}",
        f"- Failed: {sum(row['status'] == 'FAIL' for row in report)}",
        f"- Unindexed method scripts: {', '.join(unindexed) if unindexed else 'none'}",
        f"- Missing indexed scripts: {', '.join(missing_scripts) if missing_scripts else 'none'}",
        "- Checks: unique numbering, portable filename, indexed SHA-256, AST parse, and static compile.",
        "",
        "| Section | Sequence | Script | Role | Provenance | Status |",
        "|---|---:|---|---|---|---|",
    ]
    for row in report:
        lines.append(
            f"| {row['methods_section']} | {row['sequence']} | `{row['archive_filename']}` | "
            f"{row['role']} | {row['source_status']} | {row['status']} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("Analysis integrity validation failed")
    print(f"PASS: {len(report)} indexed scripts compile and match their recorded checksums")


if __name__ == "__main__":
    main()
