# Publication table package

This directory contains versioned, machine-readable tables generated from
the frozen Stage-1 split, matched cohort files, paired-timepoint manifest,
completed 20 x 5 Stage-2 manifest, and audited Tox21 assay annotations.

## Recommended manuscript allocation

### Study-flow table

Source: `table_s0_dataset_flow.csv`. This compact table reconciles the 7,462
source rows with the 5,728 scaffold-overlap exclusions, two invalid
scaffolds, and 1,732 retained Stage-1 compounds. Its LaTeX fragment is
`table_s0_dataset_flow.tex`.

### Main Table 1: Study datasets and endpoint composition

Source: `table_s1_tox21_endpoint_statistics.csv`.

Recommended columns:

- assay family and task;
- mechanistic endpoint, protocol, assay system and exposure;
- total Tox21 compounds;
- labeled, positive and negative Tox21 compounds;
- labeled, positive and negative compounds in the scaffold-excluded Stage-1
  development pool.

The supplied fragment is `table_s1_tox21_endpoint_statistics.tex`. Because the full
audit CSV also contains split-specific Stage-1 counts, retain those extra
columns for Supplementary Table S1 rather than forcing them into the main
typeset table.

### Main Table 2: Chemical candidate benchmark

Use the existing Stage-1 performance table together with
`table_s7_k4_ensemble_composition.csv`. The ensemble-composition table identifies the
four endpoint-specific constituents. Values on the Stage-1
ensemble-selection partition are development-selection statistics and must
not be described as independent post-selection generalization estimates.

### Main Table 3: Tox21 assay and LINCS configuration

Use the existing audited file:
`../main_table3_tox21_lincs_config_v1/table3_tox21_lincs_display.csv`.
The protocol metadata used here were joined from its numeric companion.

### Supplementary Table S1: Stage-1 split label statistics

Use all columns of `table_s1_tox21_endpoint_statistics.csv`. This table records train,
validation and ensemble-selection label counts separately.

### Supplementary Table S2: Primary 6-h LINCS task-cell composition

Source: `table_s2_primary_6h_task_cell_counts.csv`. Report all 48 endpoint-cell
combinations, including missing labels and active prevalence. The
`analysis_status` field marks HepG2-NR-PPAR-gamma as exploratory because it
contains fewer than 20 actives.

### Supplementary Table S3: LINCS cohort and paired-timepoint availability

Source: `table_s3_lincs_cohort_statistics.csv`. Distinguish full primary 6-h cohorts from
paired 6/24-h cohorts. HepG2 has 46 paired compounds and was excluded from
paired-timepoint model fitting.

### Supplementary Table S4: Paired-timepoint task composition

Source: `table_s4_paired_timepoint_task_cell_counts.csv`. The 6- and 24-h rows have identical
label counts by design because the same compounds and labels are used at
both exposure times.

### Supplementary Table S5: Stage-1 model hyperparameters

Source: `table_s5_stage1_model_hyperparameters.csv`. The supplied LaTeX fragment is
`table_s5_stage1_model_hyperparameters.tex`.

### Supplementary Table S6: Stage-2 and benchmark settings

Source: `table_s6_stage2_benchmark_settings.csv`. The supplied LaTeX fragment is
`table_s6_stage2_benchmark_settings.tex`.

## LaTeX generation recipe

The `.tex` fragments in this directory were generated with pandas
`DataFrame.to_latex(longtable=True, escape=True)`. To regenerate them:

```bash
python generate_tables.py \
  --output-dir _run_output/generation_regenerated_v1
```

The script refuses to overwrite an existing directory. Use a new versioned
output path for every regeneration.

Required LaTeX packages:

```latex
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{threeparttablex}
\usepackage{siunitx}
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
