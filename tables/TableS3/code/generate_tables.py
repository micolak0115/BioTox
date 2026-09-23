"""Build exact, versioned Stage-2 split and eligibility source CSVs.

Outer-fold membership is read from archived out-of-fold prediction files.
The script never reconstructs an outer split and refuses to overwrite an
existing output directory. Historical inner-fold membership was not
serialized, so no retrospective inner-fold sample counts are fabricated.
Only source-data CSVs and ``RUN_COMPLETE.json`` are written; no LaTeX,
Markdown, HTML, PDF, or page-preview rendering is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INPUT_FILES = CONFIG["input_files"]
METHODS_V4 = Path(INPUT_FILES["methods_v4"])
PRIMARY_RUN = Path(INPUT_FILES["primary_run"])
PAIRED_6H_RUN = Path(INPUT_FILES["paired_6h_run"])
DEFAULT_OUTPUT = (HERE / CONFIG["output"]["directory"]).resolve()

TASK_ORDER = [
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
]
CELL_ORDER = ["HA1E", "HEPG2", "HT29", "MCF7"]
CONFIRMATORY_MIN_TOTAL_POSITIVES = 20
MIN_OUTER_PARTITION_CLASS_COUNT = 2
PAIRED_MIN_COHORT_SIZE = 50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact Stage-2 split statistics and LaTeX tables."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New versioned output directory; existing paths are rejected.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def family(task: str) -> str:
    return "NR" if task.startswith("NR-") else "SR"


def standard_oof_paths(root: Path) -> list[Path]:
    return sorted(
        root.glob("repeated_nested_cv_oof_predictions_standardized_*.csv")
    )


def read_exact_outer_folds(root: Path, analysis: str) -> pd.DataFrame:
    paths = standard_oof_paths(root)
    if not paths:
        raise FileNotFoundError(f"No standardized OOF files found in {root}")

    rows: list[dict[str, object]] = []
    for path in paths:
        frame = pd.read_csv(
            path,
            usecols=[
                "task",
                "cell",
                "repeat_id",
                "outer_fold_id",
                "outer_split_seed",
                "outer_split_signature",
                "ik",
                "y",
            ],
        )
        if frame.empty:
            raise ValueError(f"Empty OOF file: {path}")
        task_values = frame["task"].drop_duplicates().tolist()
        cell_values = frame["cell"].drop_duplicates().tolist()
        if len(task_values) != 1 or len(cell_values) != 1:
            raise ValueError(f"Mixed task or cell in {path}")

        task = str(task_values[0])
        cell = str(cell_values[0])
        per_repeat = frame.groupby("repeat_id", sort=True)
        if per_repeat.ngroups != 20:
            raise ValueError(f"{path} does not contain 20 repeats")

        for repeat_id, repeat in per_repeat:
            if repeat["ik"].duplicated().any():
                raise ValueError(
                    f"Duplicate OOF compound in {path}, repeat {repeat_id}"
                )
            total_n = int(len(repeat))
            total_positive = int(repeat["y"].sum())
            total_negative = total_n - total_positive
            fold_groups = repeat.groupby("outer_fold_id", sort=True)
            if fold_groups.ngroups != 5:
                raise ValueError(
                    f"{path}, repeat {repeat_id} does not contain five folds"
                )
            for outer_fold_id, test in fold_groups:
                n_test = int(len(test))
                test_positive = int(test["y"].sum())
                test_negative = n_test - test_positive
                rows.append(
                    {
                        "analysis": analysis,
                        "family": family(task),
                        "task": task,
                        "cell_line": cell,
                        "repeat_id": int(repeat_id),
                        "outer_fold_id": int(outer_fold_id),
                        "outer_split_seed": int(
                            test["outer_split_seed"].iloc[0]
                        ),
                        "outer_split_signature": str(
                            test["outer_split_signature"].iloc[0]
                        ),
                        "n_labeled": total_n,
                        "n_positive": total_positive,
                        "n_negative": total_negative,
                        "n_outer_train": total_n - n_test,
                        "n_outer_train_positive": (
                            total_positive - test_positive
                        ),
                        "n_outer_train_negative": (
                            total_negative - test_negative
                        ),
                        "n_outer_test": n_test,
                        "n_outer_test_positive": test_positive,
                        "n_outer_test_negative": test_negative,
                    }
                )

    result = pd.DataFrame(rows)
    expected_groups = len(paths) * 20 * 5
    if len(result) != expected_groups:
        raise RuntimeError(
            f"Expected {expected_groups} outer-fold rows, found {len(result)}"
        )
    return result.sort_values(
        ["family", "task", "cell_line", "repeat_id", "outer_fold_id"]
    ).reset_index(drop=True)


def summarize_outer_folds(audit: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis", "family", "task", "cell_line"]
    summary = (
        audit.groupby(keys, sort=False)
        .agg(
            n_repeats=("repeat_id", "nunique"),
            n_outer_folds=("outer_fold_id", "count"),
            n_labeled=("n_labeled", "first"),
            n_positive=("n_positive", "first"),
            n_negative=("n_negative", "first"),
            min_outer_train=("n_outer_train", "min"),
            max_outer_train=("n_outer_train", "max"),
            min_outer_train_positive=("n_outer_train_positive", "min"),
            max_outer_train_positive=("n_outer_train_positive", "max"),
            min_outer_train_negative=("n_outer_train_negative", "min"),
            max_outer_train_negative=("n_outer_train_negative", "max"),
            min_outer_test=("n_outer_test", "min"),
            max_outer_test=("n_outer_test", "max"),
            min_outer_test_positive=("n_outer_test_positive", "min"),
            max_outer_test_positive=("n_outer_test_positive", "max"),
            min_outer_test_negative=("n_outer_test_negative", "min"),
            max_outer_test_negative=("n_outer_test_negative", "max"),
        )
        .reset_index()
    )
    if not (summary["n_repeats"] == 20).all():
        raise RuntimeError("Every modeled endpoint-context must have 20 repeats")
    if not (summary["n_outer_folds"] == 100).all():
        raise RuntimeError(
            "Every modeled endpoint-context must have 100 outer folds"
        )
    return summary


def active_prevalence_percent(positive: int, labeled: int) -> float:
    return 100.0 * positive / labeled if labeled else float("nan")


def eligibility_status(total_positive: int, minimum_outer_positive: int) -> str:
    if minimum_outer_positive < MIN_OUTER_PARTITION_CLASS_COUNT:
        return "Excluded"
    if total_positive < CONFIRMATORY_MIN_TOTAL_POSITIVES:
        return "Exploratory"
    return "Confirmatory-eligible"


def build_primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(METHODS_V4 / "table_s2_primary_6h_task_cell_counts.csv")
    merged = source.merge(
        summary.drop(columns=["analysis", "family"]),
        on=["task", "cell_line"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    if merged["min_outer_test_positive"].isna().any():
        raise RuntimeError("Primary OOF audit is incomplete")

    rows = []
    for record in merged.to_dict("records"):
        minimum = int(record["min_outer_test_positive"])
        rows.append(
            {
                "family": record["family"],
                "task": record["task"],
                "cell_line": record["cell_line"],
                "exposure_h": 6,
                "dose_window_uM": "8-12",
                "n_total_compounds": int(record["n_total_compounds"]),
                "n_labeled": int(record["n_labeled"]),
                "n_positive": int(record["n_positive"]),
                "n_negative": int(record["n_negative"]),
                "n_missing": int(record["n_missing"]),
                "active_class_prevalence_percent": active_prevalence_percent(
                    int(record["n_positive"]), int(record["n_labeled"])
                ),
                "outer_train_n_range": (
                    f"{int(record['min_outer_train'])}-"
                    f"{int(record['max_outer_train'])}"
                ),
                "outer_test_n_range": (
                    f"{int(record['min_outer_test'])}-"
                    f"{int(record['max_outer_test'])}"
                ),
                "outer_test_positive_range": (
                    f"{minimum}-"
                    f"{int(record['max_outer_test_positive'])}"
                ),
                "minimum_outer_test_positives": minimum,
                "eligibility": eligibility_status(
                    int(record["n_positive"]), minimum
                ),
            }
        )
    result = pd.DataFrame(rows)
    return order_task_cell(result)


def build_paired_table(summary: pd.DataFrame) -> pd.DataFrame:
    source = pd.read_csv(
        METHODS_V4 / "table_s4_paired_timepoint_task_cell_counts.csv"
    )
    source = (
        source.sort_values(["task", "cell_line", "exposure_h"])
        .drop_duplicates(["task", "cell_line"])
        .reset_index(drop=True)
    )
    modeled = source["cell_line"] != "HEPG2"
    merged = source.merge(
        summary.drop(columns=["analysis", "family"]),
        on=["task", "cell_line"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_audit"),
    )
    if merged.loc[modeled, "min_outer_test_positive"].isna().any():
        raise RuntimeError("Paired OOF audit is incomplete")
    if merged.loc[~modeled, "min_outer_test_positive"].notna().any():
        raise RuntimeError("Excluded HepG2 paired rows unexpectedly have OOF data")

    rows = []
    for record in merged.to_dict("records"):
        paired_n = int(record["paired_cohort_n_total"])
        excluded = paired_n < PAIRED_MIN_COHORT_SIZE
        if excluded:
            eligibility = "Excluded"
            train_range = "-"
            test_range = "-"
            positive_range = "-"
            minimum = pd.NA
        else:
            minimum = int(record["min_outer_test_positive"])
            eligibility = eligibility_status(
                int(record["n_positive"]), minimum
            )
            train_range = (
                f"{int(record['min_outer_train'])}-"
                f"{int(record['max_outer_train'])}"
            )
            test_range = (
                f"{int(record['min_outer_test'])}-"
                f"{int(record['max_outer_test'])}"
            )
            positive_range = (
                f"{minimum}-{int(record['max_outer_test_positive'])}"
            )
        rows.append(
            {
                "family": record["family"],
                "task": record["task"],
                "cell_line": record["cell_line"],
                "exposure_h": "6 and 24",
                "dose_window_uM": "8-12",
                "paired_cohort_n_total": paired_n,
                "n_labeled": int(record["n_labeled"]),
                "n_positive": int(record["n_positive"]),
                "n_negative": int(record["n_negative"]),
                "n_missing": int(record["n_missing"]),
                "active_class_prevalence_percent": active_prevalence_percent(
                    int(record["n_positive"]), int(record["n_labeled"])
                ),
                "outer_train_n_range": train_range,
                "outer_test_n_range": test_range,
                "outer_test_positive_range": positive_range,
                "minimum_outer_test_positives": minimum,
                "eligibility": eligibility,
            }
        )
    result = pd.DataFrame(rows)
    return order_task_cell(result)


def order_task_cell(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_task"] = pd.Categorical(
        result["task"], categories=TASK_ORDER, ordered=True
    )
    result["_cell"] = pd.Categorical(
        result["cell_line"], categories=CELL_ORDER, ordered=True
    )
    return (
        result.sort_values(["_task", "_cell"])
        .drop(columns=["_task", "_cell"])
        .reset_index(drop=True)
    )


def exact_range(values: pd.Series) -> str:
    low = int(values.min())
    high = int(values.max())
    return str(low) if low == high else f"{low}-{high}"


def build_study_design_table(
    primary_audit: pd.DataFrame,
    paired_audit: pd.DataFrame,
) -> pd.DataFrame:
    stage1 = pd.read_csv(METHODS_V4 / "table_s0_dataset_flow.csv")
    stage1 = stage1.loc[
        stage1["analysis_stage"].eq("Stage-1 chemical pool")
    ].copy()
    stage1_name = {
        "Total": "Total",
        "Training": "Training",
        "Validation": "Validation",
        "Development-selection": "Ensemble evaluation",
    }
    rows = []
    for record in stage1.to_dict("records"):
        rows.append(
            {
                "stage": "Stage 1",
                "source": "Tox21-only (scaffold-disjoint from LINCS cohorts)",
                "cell_line": "-",
                "partition": stage1_name[record["partition_or_cohort"]],
                "exact_compound_count": str(int(record["n_compounds"])),
                "fold_structure": "Fixed scaffold partition",
                "provenance": "Frozen Stage-1 split file",
            }
        )

    for label, audit, cells in [
        (
            "Stage 2 primary 6 h",
            primary_audit,
            "HA1E, HEPG2, HT29, MCF7",
        ),
        (
            "Stage 2 paired 6/24 h",
            paired_audit,
            "HA1E, HT29, MCF7",
        ),
    ]:
        rows.extend(
            [
                {
                    "stage": label,
                    "source": "LINCS-Tox21 matched compounds",
                    "cell_line": cells,
                    "partition": "Outer training",
                    "exact_compound_count": exact_range(
                        audit["n_outer_train"]
                    ),
                    "fold_structure": "20 repeats x 5 scaffold-grouped folds",
                    "provenance": "Archived OOF membership",
                },
                {
                    "stage": label,
                    "source": "LINCS-Tox21 matched compounds",
                    "cell_line": cells,
                    "partition": "Outer test",
                    "exact_compound_count": exact_range(
                        audit["n_outer_test"]
                    ),
                    "fold_structure": "20 repeats x 5 scaffold-grouped folds",
                    "provenance": "Archived OOF membership",
                },
                {
                    "stage": label,
                    "source": "LINCS-Tox21 matched compounds",
                    "cell_line": cells,
                    "partition": "Inner training",
                    "exact_compound_count": "Not serialized",
                    "fold_structure": (
                        "3 scaffold-grouped folds within outer training"
                    ),
                    "provenance": (
                        "Seed and protocol archived; membership unavailable"
                    ),
                },
                {
                    "stage": label,
                    "source": "LINCS-Tox21 matched compounds",
                    "cell_line": cells,
                    "partition": "Inner validation",
                    "exact_compound_count": "Not serialized",
                    "fold_structure": (
                        "3 scaffold-grouped folds within outer training"
                    ),
                    "provenance": (
                        "Seed and protocol archived; membership unavailable"
                    ),
                },
            ]
        )
    return pd.DataFrame(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def eligibility_latex(
    frame: pd.DataFrame,
    *,
    paired: bool,
) -> str:
    raise RuntimeError("LaTeX rendering is disabled; use the source CSV outputs.")
    columns = [
        "Family",
        "Assay",
        "Cell line",
        "Labeled",
        "Active",
        "Inactive",
        "Active-class prevalence (\\%)",
        "Outer-test n",
        "Outer-test active",
        "Eligibility",
    ]
    if paired:
        columns.insert(3, "Paired n")
    column_format = "llllrrrrlll" if paired else "lllrrrrlll"
    lines = [
        rf"\begin{{longtable}}{{{column_format}}}",
        r"\caption{Endpoint-context label composition and realized outer-test support.}\\",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
        r"\endhead",
    ]
    previous_family = None
    for record in frame.to_dict("records"):
        if previous_family is not None and record["family"] != previous_family:
            lines.append(r"\midrule\midrule")
        previous_family = record["family"]

        eligibility = record["eligibility"]
        active_support = record["outer_test_positive_range"]
        labeled = str(int(record["n_labeled"]))
        active = str(int(record["n_positive"]))
        if eligibility == "Exploratory":
            active = rf"\textbf{{{active}}}"
        cell = latex_escape(record["cell_line"])
        assay = latex_escape(record["task"])

        row = [
            latex_escape(record["family"]),
            assay,
            cell,
            labeled,
            active,
            str(int(record["n_negative"])),
            f"{record['active_class_prevalence_percent']:.1f}",
            latex_escape(record["outer_test_n_range"]),
            active_support,
            latex_escape(eligibility),
        ]
        if paired:
            paired_n = str(int(record["paired_cohort_n_total"]))
            if eligibility == "Excluded":
                paired_n = rf"\textbf{{{paired_n}}}"
            row.insert(3, paired_n)
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            "",
        ]
    )
    return "\n".join(lines)


def study_design_latex(frame: pd.DataFrame) -> str:
    raise RuntimeError("LaTeX rendering is disabled; use the source CSV outputs.")
    display = frame.rename(
        columns={
            "stage": "Stage",
            "source": "Source",
            "cell_line": "Cell line",
            "partition": "Partition",
            "exact_compound_count": "Exact compound count",
            "fold_structure": "Fold structure",
            "provenance": "Count provenance",
        }
    )
    return display.to_latex(
        index=False,
        longtable=True,
        escape=True,
        column_format="lllllll",
        caption="Study partitions and exact Stage-2 count provenance.",
        label="tab:study_design_exact_splits",
    )


def write_recipe(output: Path) -> None:
    return
    text = """# Publication split-statistics package

## Source of truth

All Stage-2 outer-fold counts are calculated from archived out-of-fold
prediction membership. They are exact realized counts, not expected 4/5 and
1/5 allocations. The detailed source is `audit_stage2_outer_fold_counts.csv`.

Historical inner-fold seeds and the three-fold protocol were archived, but
compound-level inner membership was not. Therefore, exact retrospective
inner-training and inner-validation counts are reported as `Not serialized`
rather than reconstructed under a potentially different software state.

## Files

- `table_1_study_design_exact_splits.csv` and `.tex`: Stage-1 fixed counts,
  exact realized Stage-2 outer-count ranges, and inner-count provenance.
- `table_s2_primary_6h_eligibility.csv` and `.tex`: primary 6-h endpoint-cell
  composition and realized outer-test support.
- `table_s3_paired_timepoint_eligibility.csv` and `.tex`: paired 6/24-h
  composition and realized outer-test support.
- `audit_stage2_outer_fold_counts.csv`: one exact row per repeat, outer fold,
  endpoint, and cell line.
- `audit_stage2_outer_fold_summary.csv`: exact ranges by analysis,
  endpoint, and cell line.
- `DATA_DICTIONARY.md`: definitions and decision rules.

## Eligibility rule

Eligibility is assigned per endpoint-context, not per endpoint globally.

1. **Excluded:** the paired cell-line cohort contained fewer than 50 total
   compounds before endpoint-specific filtering, or a required outer
   partition contained fewer than two active or two inactive compounds.
2. **Exploratory:** nested CV was estimable, but the endpoint-context
   contained fewer than 20 active compounds in total.
3. **Confirmatory-eligible:** the endpoint-context contained at least 20
   active compounds and every realized outer train/test partition satisfied
   the two-per-class estimability requirement.

The 20-active threshold is the prespecified, context-level interpretation
rule. Exact realized fold ranges are reported as diagnostics because
scaffold grouping can produce uneven fold sizes. The minimum over 100
repeated folds is not used as a post hoc eligibility threshold. Active-class
prevalence is descriptive and does not determine eligibility.

## LaTeX

Required packages:

```latex
\\usepackage{booktabs}
\\usepackage{longtable}
\\usepackage{threeparttablex}
\\usepackage{siunitx}
```

The supplementary fragments insert a double midrule between NR and SR
blocks. Bold active counts identify exploratory contexts; bold paired-cohort
counts identify excluded contexts.

## Regeneration

```bash
python generate_tables.py \\
  --output-dir _run_output/publication_methods_tables_v5_regenerated_v1
```

The script refuses to overwrite an existing output directory.
"""
    (output / "LATEX_RECIPE.md").write_text(text, encoding="utf-8")


def write_dictionary(output: Path) -> None:
    return
    text = """# Data dictionary and interpretation rules

## Label counts

- **Labeled:** compounds with a binary active or inactive Tox21 label.
- **Active:** labeled compounds with the positive class.
- **Inactive:** labeled compounds with the negative class.
- **Missing:** compounds with missing, inconclusive, or untested labels.
- **Active-class prevalence (%):** `100 x Active / Labeled`, equivalently
  `100 x Active / (Active + Inactive)`. Missing labels are excluded.

Active-class prevalence describes endpoint imbalance. It is not an
eligibility threshold.

## Split counts

Outer counts come from the archived OOF rows. Within each repeat, every
eligible compound occurs in exactly one outer-test fold; outer-training
counts are the complementary labeled compounds. Ranges are minima and maxima
over all 20 repeated five-fold assignments.

The historical analysis retained inner split seeds and tuning diagnostics,
but not compound-level inner membership. Exact inner counts are therefore not
available from the frozen audit record and are not estimated in these tables.

## Eligibility

- **Confirmatory-eligible:** at least 20 active compounds in the complete
  endpoint-context and at least two observations of each class in every
  realized outer-training and outer-test partition.
- **Exploratory:** fewer than 20 active compounds in the complete
  endpoint-context, while nested CV remains estimable.
- **Excluded:** paired cell-line cohort size < 50 or nested CV not estimable.

This rule is context-specific. An endpoint can be confirmatory-eligible in
one cell line and exploratory or excluded in another. Statistical evidence
still requires the prespecified positive Delta AUPRC and multiplicity-
adjusted inference; eligibility alone is not evidence of improvement.

The minimum outer-test active count is retained as a fold-support diagnostic.
It is not used as a second post hoc threshold because the minimum across 100
repeated scaffold folds is highly sensitive to one scaffold-heavy
partition.
"""
    (output / "DATA_DICTIONARY.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True)

    primary_audit = read_exact_outer_folds(PRIMARY_RUN, "primary_6h")
    paired_audit = read_exact_outer_folds(PAIRED_6H_RUN, "paired_6h_24h")
    outer_audit = pd.concat(
        [primary_audit, paired_audit], ignore_index=True
    ).sort_values(
        ["analysis", "family", "task", "cell_line", "repeat_id", "outer_fold_id"]
    )
    outer_summary = pd.concat(
        [
            summarize_outer_folds(primary_audit),
            summarize_outer_folds(paired_audit),
        ],
        ignore_index=True,
    )

    primary_table = build_primary_table(
        outer_summary.loc[outer_summary["analysis"] == "primary_6h"]
    )
    paired_table = build_paired_table(
        outer_summary.loc[
            outer_summary["analysis"] == "paired_6h_24h"
        ]
    )
    design_table = build_study_design_table(primary_audit, paired_audit)

    files = {
        "audit_stage2_outer_fold_counts.csv": outer_audit,
        "audit_stage2_outer_fold_summary.csv": outer_summary,
        "table_1_study_design_exact_splits.csv": design_table,
        "table_s2_primary_6h_eligibility.csv": primary_table,
        "table_s3_paired_timepoint_eligibility.csv": paired_table,
    }
    for name, frame in files.items():
        frame.to_csv(output / name, index=False)

    generated = sorted(path for path in output.iterdir() if path.is_file())
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "publication_split_statistics_v1",
        "output_dir": str(output),
        "outer_count_source": "archived_oof_membership",
        "outer_counts_exact": True,
        "inner_membership_serialized": False,
        "inner_counts_reported": False,
        "confirmatory_min_total_positives": CONFIRMATORY_MIN_TOTAL_POSITIVES,
        "minimum_outer_partition_class_count": (
            MIN_OUTER_PARTITION_CLASS_COUNT
        ),
        "paired_min_cohort_size": PAIRED_MIN_COHORT_SIZE,
        "primary_outer_fold_rows": int(len(primary_audit)),
        "paired_outer_fold_rows": int(len(paired_audit)),
        "generated_files": {
            path.name: sha256(path) for path in generated
        },
    }
    (output / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote publication split statistics to {output}")


if __name__ == "__main__":
    main()
