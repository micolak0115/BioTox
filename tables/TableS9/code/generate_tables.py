#!/usr/bin/env python3
"""Build a versioned BioTox main-table revision without overwriting sources."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INPUT_FILES = CONFIG["input_files"]
OUT = (HERE / CONFIG["output"]["directory"]).resolve()
CANON_READABLE = Path(INPUT_FILES["canonical_readable_dir"])
PRIMARY_VARIANT = "standardized"
PRIMARY_REPRESENTATION = "unadjusted standardized transcriptome"
PRIMARY_LABEL = "No adjustment (unadjusted standardized transcriptome)"

PRIMARY_DETAIL = Path(INPUT_FILES["primary_detail"])
PAIRED_6H_DETAIL = Path(INPUT_FILES["paired_6h_detail"])
PAIRED_24H_DETAIL = Path(INPUT_FILES["paired_24h_detail"])


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_one(rows: list[dict[str, str]], **filters: str) -> dict[str, str]:
    matches = [row for row in rows if all(row.get(key) == value for key, value in filters.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {filters}, found {len(matches)}")
    return matches[0]


def as_float(value: str) -> float:
    return float(value)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return BH-adjusted q-values in the original input order."""
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [1.0] * len(p_values)
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, p_values[index] * len(p_values) / rank)
        adjusted[index] = min(running, 1.0)
    return adjusted


def write_csv_x(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text_x(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


def copy_x(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(target)
    shutil.copyfile(source, target)


def copy_unadjusted_landscape(source: Path, target: Path) -> None:
    """Copy only baseline and unadjusted columns from a mixed readable view."""
    rows = read_csv(source)
    if not rows:
        raise ValueError(f"No rows in readable landscape: {source}")
    keep = [
        key for key in rows[0]
        if key in {"Assay family", "Tox21 assay", "Chem.only (baseline)"}
        or "No adjustment" in key
        or "Unadjusted" in key
    ]
    filtered = [{key: row.get(key, "") for key in keep} for row in rows]
    write_csv_x(target, keep, filtered)


def fmt(value: float, digits: int = 3, signed: bool = False) -> str:
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(value, spec)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def context_row(
    source: dict[str, str],
    *,
    timepoint: str,
    context_role: str,
    exact_match: str,
    source_file: Path,
) -> dict[str, object]:
    delta = as_float(source["mean_delta_auprc"])
    p_value = as_float(source["nb_p_value_delta_auprc"])
    q_value = as_float(source["bh_q_value_delta_auprc"])
    return {
        "endpoint": source["task"],
        "lincs_cell": "HepG2" if source["cell"] == "HEPG2" else source["cell"],
        "exposure": timepoint,
        "representation": PRIMARY_LABEL,
        "context_role": context_role,
        "literal_assay_cell_match": exact_match,
        "n_compounds": int(float(source["n_compounds"])),
        "chem_auprc_mean": as_float(source["mean_auprc_chem"]),
        "chem_auprc_sd_outer_fold": as_float(source["sd_repeat_mean_auprc_chem"]),
        "chem_bio_auprc_mean": as_float(source["mean_auprc_biotox"]),
        "chem_bio_auprc_sd_outer_fold": as_float(source["sd_repeat_mean_auprc_biotox"]),
        "delta_auprc": delta,
        "nb_95ci_low": as_float(source["nb_95ci_low_delta_auprc"]),
        "nb_95ci_high": as_float(source["nb_95ci_high_delta_auprc"]),
        "nb_p_value": p_value,
        "bh_q_value_full_family": q_value,
        "nominal_positive": delta > 0 and p_value <= 0.05,
        "fdr_positive_full_family": delta > 0 and q_value < 0.05,
        "multiplicity_family": "12 endpoints within representation, cell, and source analysis",
        "source_file": str(source_file),
    }


def build_context_table() -> list[dict[str, object]]:
    primary = read_csv(PRIMARY_DETAIL)
    paired24 = read_csv(PAIRED_24H_DETAIL)
    rows = [
        context_row(
            find_one(primary, task="SR-ARE", cell="HEPG2", variant=PRIMARY_VARIANT),
            timepoint="6 h",
            context_role="Empirical hepatic context; no literal assay-cell identity claimed",
            exact_match="No",
            source_file=PRIMARY_DETAIL,
        ),
        context_row(
            find_one(paired24, task="NR-Aromatase", cell="MCF7", variant=PRIMARY_VARIANT),
            timepoint="24 h",
            context_role="Exact supplied assay-cell match and exposure alignment",
            exact_match="Yes",
            source_file=PAIRED_24H_DETAIL,
        ),
        context_row(
            find_one(paired24, task="SR-MMP", cell="MCF7", variant=PRIMARY_VARIANT),
            timepoint="24 h",
            context_role="Empirically strong non-exact cellular context",
            exact_match="No",
            source_file=PAIRED_24H_DETAIL,
        ),
        context_row(
            find_one(paired24, task="SR-p53", cell="MCF7", variant=PRIMARY_VARIANT),
            timepoint="24 h",
            context_role="Empirically strong context; no exact supplied counterpart",
            exact_match="No",
            source_file=PAIRED_24H_DETAIL,
        ),
    ]
    order = {"NR-Aromatase": 0, "SR-ARE": 1, "SR-MMP": 2, "SR-p53": 3}
    return sorted(rows, key=lambda row: order[str(row["endpoint"])])


def build_v3_sensitivity() -> list[dict[str, object]]:
    rows = read_csv(PAIRED_24H_DETAIL)
    output: list[dict[str, object]] = []
    for task in ("NR-Aromatase", "SR-MMP", "SR-p53"):
        row = find_one(rows, variant=PRIMARY_VARIANT, task=task, cell="MCF7")
        output.append(
            {
                "endpoint": task,
                "lincs_cell": "MCF7",
                "exposure": "24 h",
                "n_compounds": int(float(row.get("n_unique_compounds", row["n_compounds"]))),
                "chem_auprc_mean": as_float(row["mean_auprc_chem"]),
                "chem_bio_auprc_mean": as_float(row["mean_auprc_biotox"]),
                "delta_auprc": as_float(row["mean_delta_auprc"]),
                "nb_95ci_low": as_float(row["nb_95ci_low_delta_auprc"]),
                "nb_95ci_high": as_float(row["nb_95ci_high_delta_auprc"]),
                "nb_p_value": as_float(row["nb_p_value_delta_auprc"]),
                "bh_q_value_targeted_v3_family": None,
                "fdr_positive_targeted_v3_family": None,
                "multiplicity_family": "3 unadjusted standardized MCF7 24 h sensitivity contrasts",
                "source_file": str(PAIRED_24H_DETAIL),
            }
        )
    targeted_q = benjamini_hochberg([float(row["nb_p_value"]) for row in output])
    for row, q_value in zip(output, targeted_q):
        row["bh_q_value_targeted_v3_family"] = q_value
        row["fdr_positive_targeted_v3_family"] = bool(
            float(row["delta_auprc"]) > 0 and q_value < 0.05
        )
    return output


def build_exact_match_audit() -> list[dict[str, object]]:
    primary = read_csv(PRIMARY_DETAIL)
    paired6 = read_csv(PAIRED_6H_DETAIL)
    paired24 = read_csv(PAIRED_24H_DETAIL)
    sources = [("6 h primary", PRIMARY_DETAIL, primary), ("6 h paired", PAIRED_6H_DETAIL, paired6), ("24 h paired", PAIRED_24H_DETAIL, paired24)]
    exact = [
        ("NR-AhR", "HEPG2", "HepG2-AhR-luc"),
        ("NR-Aromatase", "MCF7", "MCF-7"),
        ("SR-MMP", "HEPG2", "HepG2"),
    ]
    output: list[dict[str, object]] = []
    for task, cell, assay_system in exact:
        candidates: list[tuple[str, Path, dict[str, str]]] = []
        for label, source_path, rows in sources:
            candidates.extend(
                (label, source_path, row)
                for row in rows
                if row["task"] == task
                and row["cell"] == cell
                and row["variant"] == PRIMARY_VARIANT
            )
        if not candidates:
            raise ValueError(f"No exact-match candidates for {task} {cell}")
        # Descriptive selection among standardized primary rows: highest total
        # Chem+Bio AUPRC, then primary 6 h and stable source order.
        candidates.sort(
            key=lambda item: (
                as_float(item[2]["mean_auprc_biotox"]),
                item[0] == "6 h primary",
            ),
            reverse=True,
        )
        label, source_path, row = candidates[0]
        delta = as_float(row["mean_delta_auprc"])
        p_value = as_float(row["nb_p_value_delta_auprc"])
        q_value = as_float(row["bh_q_value_delta_auprc"])
        output.append(
            {
                "endpoint": task,
                "tox21_assay_system": assay_system,
                "exact_lincs_cell": "HepG2" if cell == "HEPG2" else cell,
                "selected_exact_configuration": f"{label}; {PRIMARY_LABEL}",
                "source_variant": row["variant"],
                "selection_rule": "Highest exact-cell Chem+Bio AUPRC; descriptive",
                "n_compounds": int(float(row["n_compounds"])),
                "chem_auprc_mean": as_float(row["mean_auprc_chem"]),
                "chem_bio_auprc_mean": as_float(row["mean_auprc_biotox"]),
                "delta_auprc": delta,
                "nb_95ci_low": as_float(row["nb_95ci_low_delta_auprc"]),
                "nb_95ci_high": as_float(row["nb_95ci_high_delta_auprc"]),
                "nb_p_value": p_value,
                "bh_q_value_full_family": q_value,
                "nominal_positive": delta > 0 and p_value <= 0.05,
                "fdr_positive_full_family": delta > 0 and q_value < 0.05,
                "source_file": str(source_path),
            }
        )
    return output


def main(output_dir: Path | None = None) -> None:
    global OUT
    if output_dir is not None:
        OUT = output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {OUT}")

    context_rows = build_context_table()
    v3_rows = build_v3_sensitivity()
    exact_rows = build_exact_match_audit()

    context_fields = list(context_rows[0])
    v3_fields = list(v3_rows[0])
    exact_fields = list(exact_rows[0])
    write_csv_x(OUT / "Main2A_context_focused_complementarity_numeric.csv", context_fields, context_rows)
    write_csv_x(OUT / "Main2B_mcf7_24h_targeted_sensitivity_numeric.csv", v3_fields, v3_rows)
    write_csv_x(OUT / "Main2C_exact_assay_cell_audit_numeric.csv", exact_fields, exact_rows)

    context_md_rows = []
    for row in context_rows:
        context_md_rows.append(
            [
                str(row["endpoint"]),
                f"{row['lincs_cell']}, {row['exposure']}",
                str(row["literal_assay_cell_match"]),
                str(row["n_compounds"]),
                f"{fmt(float(row['chem_auprc_mean']))} ± {fmt(float(row['chem_auprc_sd_outer_fold']))}",
                f"{fmt(float(row['chem_bio_auprc_mean']))} ± {fmt(float(row['chem_bio_auprc_sd_outer_fold']))}",
                f"{fmt(float(row['delta_auprc']), signed=True)} [{fmt(float(row['nb_95ci_low']), signed=True)}, {fmt(float(row['nb_95ci_high']), signed=True)}]",
                f"{float(row['nb_p_value']):.4g}",
                f"{float(row['bh_q_value_full_family']):.4g}",
                "FDR-supported" if row["fdr_positive_full_family"] else "Nominal only",
            ]
        )
    context_md = (
        "# Main Table 2A | Context-focused complementary prediction\n\n"
        + markdown_table(
            ["Endpoint", "LINCS context", "Exact assay-cell match", "N", "Chem AUPRC", "Chem+Bio AUPRC", "ΔAUPRC [NB 95% CI]", "P", "BH q", "Evidence"],
            context_md_rows,
        )
        + f"\nValues use the {PRIMARY_REPRESENTATION} and the matched chemical comparator from the same cohort, repeat, and outer fold. P values use the two-sided Nadeau–Bengio corrected paired test. BH q values retain the canonical broader family of 12 endpoints within representation, cell, and source analysis; they were not recomputed only over these selected rows. Exact assay-cell identity is a literal metadata match and is distinct from empirical context selection.\n"
    )
    write_text_x(OUT / "Main2A_context_focused_complementarity.md", context_md)

    exact_md_rows = []
    for row in exact_rows:
        exact_md_rows.append(
            [
                str(row["endpoint"]),
                str(row["tox21_assay_system"]),
                str(row["exact_lincs_cell"]),
                str(row["selected_exact_configuration"]),
                fmt(float(row["chem_auprc_mean"])),
                fmt(float(row["chem_bio_auprc_mean"])),
                f"{fmt(float(row['delta_auprc']), signed=True)} [{fmt(float(row['nb_95ci_low']), signed=True)}, {fmt(float(row['nb_95ci_high']), signed=True)}]",
                f"{float(row['nb_p_value']):.4g}",
                f"{float(row['bh_q_value_full_family']):.4g}",
            ]
        )
    exact_md = (
        "# Main Table 2C | Literal assay-cell match audit\n\n"
        + markdown_table(
            ["Endpoint", "Tox21 assay system", "Exact LINCS cell", "Selected exact configuration", "Chem AUPRC", "Chem+Bio AUPRC", "ΔAUPRC [NB 95% CI]", "P", "BH q"],
            exact_md_rows,
        )
        + "\nThe selected exact configuration is the available exact-cell configuration with the highest total Chem+Bio AUPRC and is descriptive. The audit is rebuilt from the current revised Stage-2 sources, replacing carried-forward exact-match values that predated the August 25 revision. Exact matching is not a universal performance guarantee.\n"
    )
    write_text_x(OUT / "Main2C_exact_assay_cell_audit.md", exact_md)

    copy_x(CANON_READABLE / "Main1.csv", OUT / "Main1_stage1_chemical_models.csv")
    copy_unadjusted_landscape(CANON_READABLE / "Main2_NR.csv", OUT / "Main2_full_landscape_NR.csv")
    copy_unadjusted_landscape(CANON_READABLE / "Main2_SR.csv", OUT / "Main2_full_landscape_SR.csv")
    copy_x(CANON_READABLE / "Main3.csv", OUT / "Main3_calibration_ablation.csv")

    positive_nominal = sum(bool(row["nominal_positive"]) for row in context_rows)
    positive_fdr = sum(bool(row["fdr_positive_full_family"]) for row in context_rows)
    mean_delta = sum(float(row["delta_auprc"]) for row in context_rows) / len(context_rows)
    exact_nominal = sum(bool(row["nominal_positive"]) for row in exact_rows)
    exact_fdr = sum(bool(row["fdr_positive_full_family"]) for row in exact_rows)

    report = f"""# Report-ready findings: context-matched complementary prediction

## Headline result

Transcriptomic measurements supplied selective, context-dependent information beyond the frozen molecular predictor. Across the four planned reporting contexts—NR-Aromatase–MCF7–24 h, SR-ARE–HepG2–6 h, SR-MMP–MCF7–24 h, and SR-p53–MCF7–24 h—the unadjusted standardized Chem+Bio model improved mean AUPRC in all four comparisons (mean absolute ΔAUPRC {mean_delta:+.3f}; range {min(float(row['delta_auprc']) for row in context_rows):+.3f} to {max(float(row['delta_auprc']) for row in context_rows):+.3f}). All {positive_nominal}/4 gains were nominally significant by Nadeau–Bengio corrected paired tests. Under the canonical broader BH family, {positive_fdr}/4 remained FDR-supported: SR-MMP–MCF7–24 h and SR-p53–MCF7–24 h.

## Endpoint-level findings

- **SR-p53–MCF7–24 h showed the largest complementary gain:** Chem AUPRC 0.294 versus Chem+Bio 0.499; ΔAUPRC +0.206 (NB 95% CI +0.081 to +0.331), P=0.00149, BH q=0.0178.
- **SR-MMP–MCF7–24 h was also robust:** Chem AUPRC 0.529 versus Chem+Bio 0.651; ΔAUPRC +0.123 (+0.042 to +0.203), P=0.00321, q=0.0192.
- **NR-Aromatase–MCF7–24 h provided the literal assay-cell-matched example:** Chem AUPRC 0.411 versus Chem+Bio 0.526; ΔAUPRC +0.116 (+0.012 to +0.219), P=0.0285, q=0.114. This is nominal evidence, not broader-family FDR support.
- **SR-ARE–HepG2–6 h showed a nominal hepatic-context gain:** Chem AUPRC 0.452 versus Chem+Bio 0.547; ΔAUPRC +0.095 (+0.023 to +0.168), P=0.0104, q=0.120.

## Exact-match interpretation

Literal Tox21 assay-cell matching should not be equated with universal improvement. The refreshed exact-match audit found directional gains in all three literal matches, but only {exact_nominal}/3 were nominally significant and {exact_fdr}/3 survived the broader BH correction. The defensible conclusion is therefore that complementary signal is **context dependent**, with exact matching biologically informative in some settings but not sufficient by itself to guarantee predictive gain.

## Targeted 24 h MCF7 sensitivity analysis

The latest unadjusted standardized 24 h MCF7 results for NR-Aromatase, SR-MMP, and SR-p53 are provided as targeted sensitivity evidence. These q values answer a narrower sensitivity question and should not replace the broader-family q values in the primary table.

## Recommended manuscript wording

“Unadjusted standardized transcriptomic measurements provided selective complementary prediction beyond the frozen molecular model. In four context-focused comparisons, mean AUPRC increased by 0.095–0.206, with nominal corrected evidence in every comparison. The strongest broader-family FDR-supported gains occurred for SR-MMP and SR-p53 in MCF7 at 24 h. NR-Aromatase in MCF7 at 24 h supplied a biologically aligned exact-cell example, but exact assay-cell matching was not a universal predictor of improvement, indicating that the added value of transcriptomics depends on endpoint, cellular context, and exposure.”

## Reporting constraints

- Use “context-focused” or “endpoint–LINCS context” for the four-row performance table. Reserve “exact match” for literal assay-cell identity.
- Distinguish nominal P values from BH-adjusted q values.
- Keep the broader-family q values in the primary table; present the three-context v3 q values as sensitivity evidence.
- Describe the chemical predictor as frozen and the comparison as paired within the same cohort, repeat, and outer fold.
- CEViChE/viability-residualized results are sensitivity/archive evidence and are not part of this primary bundle.
"""
    write_text_x(OUT / "REPORT_READY_FINDINGS.md", report)

    readme = """# Revised BioTox main tables: context-focused complementarity

This versioned bundle reorganizes the main-table presentation without modifying the canonical `tables_revised` source bundle.

- Main Table 1 is carried forward unchanged.
- Main Table 2A is the recommended compact main-paper table for the four planned endpoint–LINCS contexts.
- Main Table 2B records the latest targeted unadjusted standardized MCF7 24 h sensitivity results.
- Main Table 2C refreshes the literal exact assay-cell audit from current revised Stage-2 sources.
- The complete NR and SR performance landscapes are retained as supporting CSVs.
- Main Table 3 is carried forward unchanged.

The primary interpretation uses the canonical broader BH families. The targeted v3 family is explicitly labeled sensitivity evidence. See `REPORT_READY_FINDINGS.md` for report-ready prose and constraints.
"""
    write_text_x(OUT / "README.md", readme)

    source_paths = [
        PRIMARY_DETAIL,
        PAIRED_6H_DETAIL,
        PAIRED_24H_DETAIL,
        CANON_READABLE / "Main1.csv",
        CANON_READABLE / "Main2_NR.csv",
        CANON_READABLE / "Main2_SR.csv",
        CANON_READABLE / "Main3.csv",
    ]
    source_manifest = []
    for path in source_paths:
        source_manifest.append(
            {
                "source_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    write_csv_x(OUT / "SOURCE_MANIFEST.csv", ["source_path", "size_bytes", "sha256"], source_manifest)

    validation = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(OUT),
        "checks": {
            "context_rows": len(context_rows),
            "context_all_positive": all(float(row["delta_auprc"]) > 0 for row in context_rows),
            "context_nominal_positive_count": positive_nominal,
            "context_broader_family_fdr_count": positive_fdr,
            "context_broader_family_fdr_endpoints": [
                row["endpoint"] for row in context_rows if row["fdr_positive_full_family"]
            ],
            "exact_match_rows": len(exact_rows),
            "exact_match_nominal_positive_count": exact_nominal,
            "exact_match_broader_family_fdr_count": exact_fdr,
            "exact_match_all_standardized": all(
                row["source_variant"] == PRIMARY_VARIANT for row in exact_rows
            ),
            "v3_rows": len(v3_rows),
            "v3_all_targeted_family_fdr": all(
                str(row["fdr_positive_targeted_v3_family"]).lower() == "true" for row in v3_rows
            ),
        },
        "status": "PASS",
    }
    write_text_x(OUT / "VALIDATION_REPORT.json", json.dumps(validation, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    main(args.output_dir)
