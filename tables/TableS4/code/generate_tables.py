"""Build versioned dataset and hyperparameter tables for the manuscript.

The script reads only frozen inputs and completed-run manifests. It refuses
to overwrite an existing output directory.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
TABLES_ROOT = PROJECT_ROOT / "tables"
TABLE_NAME = "TableS4"
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage1.splitter import generate_scaffold  # noqa: E402


TASKS = [
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
CELLS = ["HA1E", "HEPG2", "HT29", "MCF7"]

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
INPUT_FILES = CONFIG["input_files"]
TOX21_PATH = Path(INPUT_FILES["tox21"])
COHORT_DIR = Path(INPUT_FILES["cohort_dir"])
PAIRED_MANIFEST = Path(INPUT_FILES["paired_manifest"])
PRIMARY_RUN_MANIFEST = Path(INPUT_FILES["primary_run_manifest"])
ASSAY_METADATA = Path(INPUT_FILES["assay_metadata"])
ENSEMBLE_COMPOSITION = Path(INPUT_FILES["ensemble_composition"])
DEFAULT_OUTPUT = (HERE / CONFIG["output"]["directory"]).resolve()


def load_split(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def combine_split(split) -> pd.DataFrame:
    return pd.concat(
        [split.train, split.valid, split.test],
        axis=0,
        ignore_index=True,
    )


def label_counts(frame: pd.DataFrame, task: str) -> dict[str, float | int]:
    values = pd.to_numeric(frame[task], errors="coerce")
    labeled = values.dropna()
    positive = int((labeled == 1).sum())
    negative = int((labeled == 0).sum())
    return {
        "n_total_compounds": int(len(frame)),
        "n_labeled": int(len(labeled)),
        "n_positive": positive,
        "n_negative": negative,
        "n_missing": int(values.isna().sum()),
        "positive_prevalence": (
            float(positive / len(labeled)) if len(labeled) else float("nan")
        ),
    }


def prefixed_counts(
    frame: pd.DataFrame,
    task: str,
    prefix: str,
) -> dict[str, float | int]:
    return {
        f"{prefix}_{key}": value
        for key, value in label_counts(frame, task).items()
    }


def assay_metadata_table() -> pd.DataFrame:
    metadata = pd.read_csv(ASSAY_METADATA)
    keep = [
        "family",
        "task",
        "mechanistic_endpoint",
        "tox21_protocol",
        "tox21_assay_system",
        "tox21_exposure_h",
        "tox21_vehicle",
        "tox21_n_concentration_points",
        "exact_lincs_cell",
    ]
    return metadata[keep].copy()


def build_tox21_endpoint_statistics() -> pd.DataFrame:
    tox21 = pd.read_csv(TOX21_PATH)
    stage1 = load_split(
        COHORT_DIR / "tox21_scaffold_df_split_chem_finetune_pool.pkl"
    )
    stage1_all = combine_split(stage1)
    metadata = assay_metadata_table().set_index("task")

    rows = []
    for task in TASKS:
        row = metadata.loc[task].to_dict()
        row["task"] = task
        row.update(prefixed_counts(tox21, task, "tox21_source"))
        row.update(prefixed_counts(stage1_all, task, "stage1_pool"))
        row.update(prefixed_counts(stage1.train, task, "stage1_train"))
        row.update(prefixed_counts(stage1.valid, task, "stage1_validation"))
        row.update(
            prefixed_counts(stage1.test, task, "stage1_ensemble_selection")
        )
        rows.append(row)

    columns = ["family", "task"] + [
        column
        for column in rows[0]
        if column not in {"family", "task"}
    ]
    return pd.DataFrame(rows)[columns]


def build_dataset_flow() -> pd.DataFrame:
    tox21 = pd.read_csv(TOX21_PATH)
    scaffolds = tox21["smiles_canon"].map(
        lambda value: generate_scaffold(value, include_chirality=True)
    )
    stage1 = load_split(
        COHORT_DIR / "tox21_scaffold_df_split_chem_finetune_pool.pkl"
    )
    n_source = int(len(tox21))
    n_invalid = int(scaffolds.isna().sum())
    n_stage1 = int(len(stage1.train) + len(stage1.valid) + len(stage1.test))
    n_overlap = n_source - n_invalid - n_stage1
    return pd.DataFrame(
        [
            {
                "analysis_stage": "Tox21 source",
                "partition_or_cohort": "All source rows",
                "n_compounds": n_source,
                "role": "Starting chemical and endpoint-label collection",
            },
            {
                "analysis_stage": "Stage-1 exclusion",
                "partition_or_cohort": "LINCS-overlapping scaffolds",
                "n_compounds": n_overlap,
                "role": "Excluded before chemical-prior development",
            },
            {
                "analysis_stage": "Stage-1 exclusion",
                "partition_or_cohort": "Invalid or unavailable scaffold",
                "n_compounds": n_invalid,
                "role": "Excluded before chemical-prior development",
            },
            {
                "analysis_stage": "Stage-1 chemical pool",
                "partition_or_cohort": "Total",
                "n_compounds": n_stage1,
                "role": "Scaffold-disjoint chemical-prior development",
            },
            {
                "analysis_stage": "Stage-1 chemical pool",
                "partition_or_cohort": "Training",
                "n_compounds": int(len(stage1.train)),
                "role": "Candidate parameter estimation",
            },
            {
                "analysis_stage": "Stage-1 chemical pool",
                "partition_or_cohort": "Validation",
                "n_compounds": int(len(stage1.valid)),
                "role": "Within-candidate tuning and checkpoint selection",
            },
            {
                "analysis_stage": "Stage-1 chemical pool",
                "partition_or_cohort": "Development-selection",
                "n_compounds": int(len(stage1.test)),
                "role": "Candidate ranking and ensemble-size selection",
            },
        ]
    )


def build_primary_6h_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows = []
    cohort_rows = []
    all_compounds = set()
    for cell in CELLS:
        path = COHORT_DIR / f"tox21_scaffold_df_split_{cell}_8to12uM_6h.pkl"
        frame = combine_split(load_split(path))
        all_compounds.update(frame["ik"].astype(str))
        scaffolds = {
            generate_scaffold(smiles, include_chirality=True)
            for smiles in frame["smiles_canon"].astype(str)
        }
        scaffolds.discard(None)
        cohort_rows.append(
            {
                "analysis": "primary_6h",
                "cell_line": cell,
                "exposure_h": 6,
                "dose_window_uM": "8-12",
                "n_compounds": int(len(frame)),
                "n_unique_scaffolds": int(len(scaffolds)),
                "n_landmark_genes": 978,
                "replicate_aggregation": "feature-wise median",
                "included_in_named_analysis": True,
                "analysis_scope_note": "Included in the complete primary 6 h analysis",
            }
        )
        for task in TASKS:
            row = {
                "family": task.split("-", 1)[0],
                "task": task,
                "cell_line": cell,
                "exposure_h": 6,
                "dose_window_uM": "8-12",
            }
            row.update(label_counts(frame, task))
            row["confirmatory_active_count_eligible"] = (
                row["n_positive"] >= 20
            )
            row["analysis_status"] = (
                "confirmatory"
                if row["confirmatory_active_count_eligible"]
                else "exploratory"
            )
            task_rows.append(row)

    cohort_rows.append(
        {
            "analysis": "primary_6h_union",
            "cell_line": "Union",
            "exposure_h": 6,
            "dose_window_uM": "8-12",
            "n_compounds": int(len(all_compounds)),
            "n_unique_scaffolds": np.nan,
            "n_landmark_genes": 978,
            "replicate_aggregation": "feature-wise median",
            "included_in_named_analysis": True,
            "analysis_scope_note": "Union of complete primary 6 h cell-line cohorts",
        }
    )
    return pd.DataFrame(task_rows), pd.DataFrame(cohort_rows)


def build_paired_timepoint_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = json.loads(PAIRED_MANIFEST.read_text())
    cohort_rows = []
    task_rows = []
    for cell in CELLS:
        record = manifest["cells"][cell]
        included = cell != "HEPG2"
        cohort_rows.extend(
            [
                {
                    "analysis": "all_available_timepoint",
                    "cell_line": cell,
                    "exposure_h": 6,
                    "dose_window_uM": "8-12",
                    "n_compounds": int(record["n_6h"]),
                    "n_unique_scaffolds": np.nan,
                    "n_landmark_genes": int(manifest["n_gene_features"]),
                    "replicate_aggregation": manifest[
                        "replicate_aggregation"
                    ],
                    "included_in_named_analysis": True,
                    "analysis_scope_note": (
                        "Included in the complete primary 6 h analysis; "
                        "this flag does not denote paired-timepoint eligibility"
                    ),
                },
                {
                    "analysis": "all_available_timepoint",
                    "cell_line": cell,
                    "exposure_h": 24,
                    "dose_window_uM": "8-12",
                    "n_compounds": int(record["n_24h"]),
                    "n_unique_scaffolds": np.nan,
                    "n_landmark_genes": int(manifest["n_gene_features"]),
                    "replicate_aggregation": manifest[
                        "replicate_aggregation"
                    ],
                    "included_in_named_analysis": False,
                    "analysis_scope_note": (
                        "Descriptive availability count; paired analysis "
                        "uses only compounds observed at both times"
                    ),
                },
                {
                    "analysis": "paired_6h_24h",
                    "cell_line": cell,
                    "exposure_h": "6 and 24",
                    "dose_window_uM": "8-12",
                    "n_compounds": int(record["n_paired"]),
                    "n_unique_scaffolds": np.nan,
                    "n_landmark_genes": int(manifest["n_gene_features"]),
                    "replicate_aggregation": manifest[
                        "replicate_aggregation"
                    ],
                    "included_in_named_analysis": included,
                    "analysis_scope_note": (
                        ""
                        if included
                        else "Excluded from paired modeling: only 46 compounds"
                    ),
                },
            ]
        )
        for task in TASKS:
            counts = record["task_counts"][task]
            for time_h in (6, 24):
                task_rows.append(
                    {
                        "family": task.split("-", 1)[0],
                        "task": task,
                        "cell_line": cell,
                        "exposure_h": time_h,
                        "paired_cohort_n_total": int(record["n_paired"]),
                        "n_labeled": int(counts["n_labeled"]),
                        "n_missing": int(
                            record["n_paired"] - counts["n_labeled"]
                        ),
                        "n_positive": int(counts["n_positive"]),
                        "n_negative": int(counts["n_negative"]),
                        "positive_prevalence": float(
                            counts["n_positive"] / counts["n_labeled"]
                        ),
                        "paired_model_included": included,
                        "analysis_status": (
                            "excluded_from_paired_modeling"
                            if not included
                            else (
                                "confirmatory"
                                if counts["n_positive"] >= 20
                                else "exploratory_active_count_below_20"
                            )
                        ),
                    }
                )
    return pd.DataFrame(task_rows), pd.DataFrame(cohort_rows)


def build_stage1_hyperparameters() -> pd.DataFrame:
    source_classical = "stage1/classical_baseline.py"
    source_stage1 = "publication/chem_stage1_per_task_offset.py"
    source_chemprop = "stage1/chemprop_stage1_candidate.py"
    rows = [
        {
            "candidate": "MACCS_LR",
            "representation": "MACCS structural keys",
            "representation_size": 167,
            "model": "Logistic regression",
            "optimization_or_tuning": (
                "C in {0.001, 0.01, 0.1, 1, 10, 100}; selected by "
                "validation AUPRC"
            ),
            "class_imbalance": "class_weight=balanced",
            "fixed_settings": "max_iter=2000",
            "training_seed": 1,
            "source": source_classical,
        },
        {
            "candidate": "Morgan_LR",
            "representation": "Morgan fingerprint",
            "representation_size": 2048,
            "model": "Logistic regression",
            "optimization_or_tuning": (
                "radius=2; C in {0.001, 0.01, 0.1, 1, 10, 100}; "
                "selected by validation AUPRC"
            ),
            "class_imbalance": "class_weight=balanced",
            "fixed_settings": "max_iter=2000",
            "training_seed": 1,
            "source": source_classical,
        },
        {
            "candidate": "RDKit_LR",
            "representation": "RDKit topological fingerprint",
            "representation_size": 2048,
            "model": "Logistic regression",
            "optimization_or_tuning": (
                "C in {0.001, 0.01, 0.1, 1, 10, 100}; selected by "
                "validation AUPRC"
            ),
            "class_imbalance": "class_weight=balanced",
            "fixed_settings": "max_iter=2000",
            "training_seed": 1,
            "source": source_classical,
        },
    ]
    for fingerprint in ("MACCS", "Morgan", "RDKit"):
        size = 167 if fingerprint == "MACCS" else 2048
        extra = "radius=2; " if fingerprint == "Morgan" else ""
        rows.append(
            {
                "candidate": f"{fingerprint}_XGB",
                "representation": (
                    "MACCS structural keys"
                    if fingerprint == "MACCS"
                    else (
                        "Morgan fingerprint"
                        if fingerprint == "Morgan"
                        else "RDKit topological fingerprint"
                    )
                ),
                "representation_size": size,
                "model": "XGBoost",
                "optimization_or_tuning": (
                    f"{extra}max_depth in {{3, 4, 6}}; learning_rate in "
                    "{0.05, 0.1}; selected by validation AUPRC"
                ),
                "class_imbalance": "scale_pos_weight=n_negative/n_positive",
                "fixed_settings": (
                    "n_estimators=300; subsample=0.8; "
                    "colsample_bytree=0.8; early_stopping_rounds=20; "
                    "eval_metric=aucpr; n_jobs=1"
                ),
                "training_seed": 1,
                "source": source_classical,
            }
        )
    for architecture in ("GCN", "GAT", "GIN", "GraphSAGE"):
        rows.append(
            {
                "candidate": f"{architecture}_Pretrained",
                "representation": "Molecular graph",
                "representation_size": 300,
                "model": architecture,
                "optimization_or_tuning": (
                    "Initialized from supervised_contextpred checkpoint; "
                    "end-to-end single-task fine-tuning; validation AUPRC "
                    "checkpoint selection"
                ),
                "class_imbalance": "BCE pos_weight=n_negative/n_positive",
                "fixed_settings": (
                    "5 message-passing layers; hidden_dim=300; "
                    "mean pooling; JK=last; dropout=0.5; batch_size=32; "
                    "AdamW lr=1e-4; max_epochs=100; patience=20"
                ),
                "training_seed": 1,
                "source": source_stage1,
            }
        )
    rows.extend(
        [
            {
                "candidate": "ChemBERTa_Pretrained",
                "representation": "Canonical SMILES tokens",
                "representation_size": np.nan,
                "model": "DeepChem/ChemBERTa-100M-MLM",
                "optimization_or_tuning": (
                    "End-to-end single-task fine-tuning; validation AUPRC "
                    "checkpoint selection"
                ),
                "class_imbalance": "BCE pos_weight=n_negative/n_positive",
                "fixed_settings": (
                    "max_length=128; batch_size=32; AdamW lr=1e-4; "
                    "weight_decay=0; linear warmup=6%; "
                    "gradient clipping=1.0; max_epochs=20; patience=5"
                ),
                "training_seed": 1,
                "source": source_stage1,
            },
            {
                "candidate": "Chemprop",
                "representation": "Directed molecular graph",
                "representation_size": 300,
                "model": "Chemprop D-MPNN",
                "optimization_or_tuning": (
                    "Single-task training; early stopping on validation loss"
                ),
                "class_imbalance": "Unweighted binary classification loss",
                "fixed_settings": (
                    "hidden_dim=300; depth=3; mean aggregation; "
                    "dropout=0; batch_size=32; max_epochs=100; patience=20"
                ),
                "training_seed": 1,
                "source": source_chemprop,
            },
        ]
    )
    return pd.DataFrame(rows)


def build_stage2_settings() -> pd.DataFrame:
    manifest = json.loads(PRIMARY_RUN_MANIFEST.read_text())
    alpha_grid = manifest["alpha_grid"]
    settings = [
        ("Study design", "Chemical prior", "Endpoint-specific top-4 arithmetic mean"),
        ("Study design", "Chemical prior coefficient", "Fixed at 1.0"),
        ("Study design", "Stage-2 intercept", "Unpenalized and fit within each training partition"),
        ("Study design", "Transcriptomic coefficients", "L2-regularized ridge coefficients"),
        ("Input", "Cell lines", ", ".join(manifest["cells"])),
        ("Input", "Primary exposure", "6 h"),
        ("Input", "Dose window", "8-12 uM"),
        ("Input", "Genes", "978 L1000 landmark genes"),
        ("Input", "Replicate aggregation", "Feature-wise median"),
        ("Preprocessing", "No-adjustment variant", "Fold-local z-standardization"),
        (
            "Preprocessing",
            "Viability-adjusted variant",
            "Projection away from fixed external CeViChe direction; no additional z-scaling",
        ),
        ("Validation", "Outer repeats", str(manifest["n_repeats"])),
        ("Validation", "Outer folds per repeat", str(manifest["outer_folds_per_repeat"])),
        ("Validation", "Outer grouping", "Bemis-Murcko scaffold"),
        ("Validation", "Distinct outer partitions", "Required; duplicate signatures forbidden"),
        ("Validation", "Minimum class count", "At least 2 positives and 2 negatives in outer train and test"),
        ("Validation", "Inner folds", str(manifest["inner_folds"])),
        ("Validation", "Inner grouping", "Stratified Bemis-Murcko scaffold groups"),
        ("Validation", "Inner selection metric", "Mean binomial deviance"),
        ("Validation", "Ridge selection rule", "One-standard-error rule favoring stronger regularization"),
        (
            "Optimization",
            "Ridge alpha grid",
            f"{len(alpha_grid)} values, 10^-4 to 10^8 at 2 points per decade",
        ),
        ("Optimization", "Solver", "L-BFGS-B with analytical gradient"),
        ("Optimization", "Primary max iterations", "500"),
        ("Optimization", "Cold-retry max iterations", "2000"),
        ("Optimization", "Gradient tolerance", "1e-6"),
        ("Optimization", "Acceptance gradient", "1e-5"),
        ("Optimization", "Function tolerance", "1e-14"),
        ("Likelihood", "Primary objective", "Unweighted Bernoulli negative log-likelihood"),
        ("Metric", "Primary discrimination metric", "AUPRC"),
        ("Metric", "Secondary discrimination metric", "AUROC"),
        ("Metric", "Calibration metrics", "Log loss and Brier score"),
        (
            "Inference",
            "Paired effect",
            "Combined minus recalibrated chemical model on identical outer-test compounds",
        ),
        (
            "Inference",
            "Dependence correction",
            "Nadeau-Bengio corrected resampled paired t-test on 100 outer-fold effects",
        ),
        (
            "Inference",
            "Multiple testing",
            "Benjamini-Hochberg within variant and cell across 12 endpoints",
        ),
        (
            "Interpretation",
            "Confirmatory active-count threshold",
            "At least 20 active compounds per endpoint-cell cohort",
        ),
        ("Reproducibility", "Stage-1 seed", "1"),
        (
            "Reproducibility",
            "Outer seed requests",
            ", ".join(str(value) for value in manifest["outer_seed_requests"]),
        ),
        ("Reproducibility", "Successful Stage-2 outer fits", str(manifest["n_successful_outer_folds"])),
        ("Reproducibility", "Failed Stage-2 outer fits", str(manifest["n_failed_outer_folds"])),
    ]
    frame = pd.DataFrame(settings, columns=["section", "setting", "value"])
    code_settings = {
        "Solver",
        "Primary max iterations",
        "Cold-retry max iterations",
        "Gradient tolerance",
        "Acceptance gradient",
        "Function tolerance",
    }
    frame["provenance"] = np.where(
        frame["setting"].isin(code_settings),
        (
            "Versioned executable defaults in offset_ridge_lbfgs.py and "
            "nested_cv_offset_logistic_calibrated_v2.py; not persisted in "
            "the completed-run manifest"
        ),
        "Completed-run manifest and archived audit artifacts",
    )
    return frame


def build_ensemble_composition() -> pd.DataFrame:
    """Load the frozen K=4 constituents and correct display-only terminology."""
    composition = pd.read_csv(ENSEMBLE_COMPOSITION)
    composition["model"] = composition["model"].str.replace(
        "RDKit descriptors",
        "RDKit topological fingerprint",
        regex=False,
    )
    return composition


def write_latex_fragment(
    frame: pd.DataFrame,
    path: Path,
    *,
    caption: str,
    label: str,
    columns: list[str],
) -> None:
    display = frame[columns].copy()
    display.columns = [column.replace("_", " ") for column in display.columns]
    latex = display.to_latex(
        index=False,
        longtable=True,
        escape=True,
        caption=caption,
        label=label,
        na_rep="NA",
        float_format=lambda value: f"{value:.4g}",
    )
    path.write_text(latex)


def write_recipe(output_dir: Path, files: dict[str, Path]) -> None:
    recipe = f"""# Publication table package

This directory contains versioned, machine-readable tables generated from
the frozen Stage-1 split, matched cohort files, paired-timepoint manifest,
completed 20 x 5 Stage-2 manifest, and audited Tox21 assay annotations.

## Recommended manuscript allocation

### Study-flow table

Source: `{files['flow'].name}`. This compact table reconciles the 7,462
source rows with the 5,728 scaffold-overlap exclusions, two invalid
scaffolds, and 1,732 retained Stage-1 compounds. Its LaTeX fragment is
`{files['flow_tex'].name}`.

### Main Table 1: Study datasets and endpoint composition

Source: `{files['endpoint'].name}`.

Recommended columns:

- assay family and task;
- mechanistic endpoint, protocol, assay system and exposure;
- total Tox21 compounds;
- labeled, positive and negative Tox21 compounds;
- labeled, positive and negative compounds in the scaffold-excluded Stage-1
  development pool.

The supplied fragment is `{files['endpoint_tex'].name}`. Because the full
audit CSV also contains split-specific Stage-1 counts, retain those extra
columns for Supplementary Table S1 rather than forcing them into the main
typeset table.

### Main Table 2: Chemical candidate benchmark

Use the existing Stage-1 performance table together with
`{files['ensemble'].name}`. The ensemble-composition table identifies the
four endpoint-specific constituents. Values on the Stage-1
ensemble-selection partition are development-selection statistics and must
not be described as independent post-selection generalization estimates.

### Main Table 3: Tox21 assay and LINCS configuration

Use the existing audited file:
`../main_table3_tox21_lincs_config_v1/table3_tox21_lincs_display.csv`.
The protocol metadata used here were joined from its numeric companion.

### Supplementary Table S1: Stage-1 split label statistics

Use all columns of `{files['endpoint'].name}`. This table records train,
validation and ensemble-selection label counts separately.

### Supplementary Table S2: Primary 6-h LINCS task-cell composition

Source: `{files['primary_counts'].name}`. Report all 48 endpoint-cell
combinations, including missing labels and active prevalence. The
`analysis_status` field marks HepG2-NR-PPAR-gamma as exploratory because it
contains fewer than 20 actives.

### Supplementary Table S3: LINCS cohort and paired-timepoint availability

Source: `{files['cohort'].name}`. Distinguish full primary 6-h cohorts from
paired 6/24-h cohorts. HepG2 has 46 paired compounds and was excluded from
paired-timepoint model fitting.

### Supplementary Table S4: Paired-timepoint task composition

Source: `{files['paired_counts'].name}`. The 6- and 24-h rows have identical
label counts by design because the same compounds and labels are used at
both exposure times.

### Supplementary Table S5: Stage-1 model hyperparameters

Source: `{files['stage1_hyper'].name}`. The supplied LaTeX fragment is
`{files['stage1_hyper_tex'].name}`.

### Supplementary Table S6: Stage-2 and benchmark settings

Source: `{files['stage2_settings'].name}`. The supplied LaTeX fragment is
`{files['stage2_settings_tex'].name}`.

## LaTeX generation recipe

The `.tex` fragments in this directory were generated with pandas
`DataFrame.to_latex(longtable=True, escape=True)`. To regenerate them:

```bash
python generate_tables.py \\
  --output-dir _run_output/{output_dir.name}_regenerated_v1
```

The script refuses to overwrite an existing directory. Use a new versioned
output path for every regeneration.

Required LaTeX packages:

```latex
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{threeparttablex}}
\\usepackage{{siunitx}}
```

For Oxford journal typesetting, keep the audit CSVs as the numerical source
of truth and shorten wide main-table columns during manuscript assembly.
Do not recompute counts or statistics from rounded display strings.

## Table notes

- AUPRC is the primary discrimination metric; AUROC is secondary.
- Missing, inconclusive and untested endpoint labels are omitted from model
  fitting and performance evaluation.
- Stage 1 uses one fixed seed and a scaffold-disjoint chemical-development
  pool.
- Stage 2 uses 20 repeated five-fold outer scaffold partitions and
  three-fold inner scaffold-grouped tuning.
- Absolute metric SDs describe resampling variability; they are not
  confidence intervals.
- Paired effect confidence intervals and P values use the Nadeau-Bengio
  correction; q values use Benjamini-Hochberg correction within
  representation and cell line across 12 endpoints.
"""
    (output_dir / "LATEX_RECIPE.md").write_text(recipe)


def main(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=False)

    flow = build_dataset_flow()
    endpoint = build_tox21_endpoint_statistics()
    primary_counts, primary_cohorts = build_primary_6h_counts()
    paired_counts, paired_cohorts = build_paired_timepoint_tables()
    cohorts = pd.concat(
        [primary_cohorts, paired_cohorts],
        axis=0,
        ignore_index=True,
    )
    stage1_hyper = build_stage1_hyperparameters()
    stage2_settings = build_stage2_settings()
    ensemble = build_ensemble_composition()

    paths = {
        "flow": output_dir / "table_s0_dataset_flow.csv",
        "endpoint": output_dir / "table_s1_tox21_endpoint_statistics.csv",
        "primary_counts": output_dir / "table_s2_primary_6h_task_cell_counts.csv",
        "cohort": output_dir / "table_s3_lincs_cohort_statistics.csv",
        "paired_counts": output_dir / "table_s4_paired_timepoint_task_cell_counts.csv",
        "stage1_hyper": output_dir / "table_s5_stage1_model_hyperparameters.csv",
        "stage2_settings": output_dir / "table_s6_stage2_benchmark_settings.csv",
        "ensemble": output_dir / "table_s7_k4_ensemble_composition.csv",
        "flow_tex": output_dir / "table_s0_dataset_flow.tex",
        "endpoint_tex": output_dir / "table_s1_tox21_endpoint_statistics.tex",
        "stage1_hyper_tex": output_dir / "table_s5_stage1_model_hyperparameters.tex",
        "stage2_settings_tex": output_dir / "table_s6_stage2_benchmark_settings.tex",
    }
    flow.to_csv(paths["flow"], index=False)
    endpoint.to_csv(paths["endpoint"], index=False)
    primary_counts.to_csv(paths["primary_counts"], index=False)
    cohorts.to_csv(paths["cohort"], index=False)
    paired_counts.to_csv(paths["paired_counts"], index=False)
    stage1_hyper.to_csv(paths["stage1_hyper"], index=False)
    stage2_settings.to_csv(paths["stage2_settings"], index=False)
    ensemble.to_csv(paths["ensemble"], index=False)

    write_latex_fragment(
        flow,
        paths["flow_tex"],
        caption="Dataset flow and Stage-1 chemical-development partitions.",
        label="tab:dataset_flow",
        columns=[
            "analysis_stage",
            "partition_or_cohort",
            "n_compounds",
            "role",
        ],
    )
    write_latex_fragment(
        endpoint,
        paths["endpoint_tex"],
        caption="Tox21 assay definitions and dataset composition.",
        label="tab:tox21_dataset_statistics",
        columns=[
            "family",
            "task",
            "mechanistic_endpoint",
            "tox21_protocol",
            "tox21_assay_system",
            "tox21_exposure_h",
            "tox21_source_n_total_compounds",
            "tox21_source_n_labeled",
            "tox21_source_n_positive",
            "tox21_source_n_negative",
            "stage1_pool_n_labeled",
            "stage1_pool_n_positive",
            "stage1_pool_n_negative",
        ],
    )
    write_latex_fragment(
        stage1_hyper,
        paths["stage1_hyper_tex"],
        caption="Stage-1 chemical candidate hyperparameters.",
        label="tab:stage1_hyperparameters",
        columns=[
            "candidate",
            "representation",
            "representation_size",
            "model",
            "optimization_or_tuning",
            "class_imbalance",
            "fixed_settings",
        ],
    )
    write_latex_fragment(
        stage2_settings,
        paths["stage2_settings_tex"],
        caption="Stage-2 model, validation and inference settings.",
        label="tab:stage2_settings",
        columns=["section", "setting", "value"],
    )
    write_recipe(output_dir, paths)

    manifest = {
        "status": "complete",
        "generator": str(Path(__file__).resolve()),
        "source_files": {
            "tox21": str(TOX21_PATH),
            "cohort_dir": str(COHORT_DIR),
            "paired_manifest": str(PAIRED_MANIFEST),
            "primary_run_manifest": str(PRIMARY_RUN_MANIFEST),
            "assay_metadata": str(ASSAY_METADATA),
            "ensemble_composition": str(ENSEMBLE_COMPOSITION),
        },
        "outputs": {
            key: str(path)
            for key, path in paths.items()
        },
        "row_counts": {
            "dataset_flow": int(len(flow)),
            "tox21_endpoint_statistics": int(len(endpoint)),
            "primary_6h_task_cell_counts": int(len(primary_counts)),
            "lincs_cohort_statistics": int(len(cohorts)),
            "paired_timepoint_task_cell_counts": int(len(paired_counts)),
            "stage1_model_hyperparameters": int(len(stage1_hyper)),
            "stage2_benchmark_settings": int(len(stage2_settings)),
            "k4_ensemble_constituents": int(len(ensemble)),
        },
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(f"Wrote publication table package to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    args = parser.parse_args()
    main(args.output_dir)
