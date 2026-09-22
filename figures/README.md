# Final figures

Each figure directory contains the publication asset, frozen source data, and
its canonical generator at `code/generate_figure.py`. Figure 2 and Figure 4
also retain the helper modules required by their generators.

Figure 1 is materialized from `Figure1/source_data/Figure1_editable.svg` by
`Figure1/code/generate_figure.py`.

## Manuscript cross-check

The manuscript used for the comparison is:

`/home/kyungan/scripts/BioTox/publication/manuscript/BioTox_BiB_Submission_20260921.tex`

The following publication source-data files were compared with the values,
statistics, captions, and tables in the manuscript and were judged to match.
These are the files consumed by the canonical figure generators, not merely
the rendered SVG/PDF assets.

| Figure | Matched source-data filepath |
|---|---|
| Figure 2 | `/data/kyungan/scripts/BioTox/publication/figures/Figure2/source_data/Figure2_topk_ensemble_curve.csv` |
| Figure 2 alias | `/data/kyungan/scripts/BioTox/publication/figures/Figure2/source_data/Figure2_ensemble_size_selection.csv` |
| Figure 3 | `/data/kyungan/scripts/BioTox/publication/figures/Figure3/source_data/Figure3_all_context_inference.csv` |
| Figure 3 | `/data/kyungan/scripts/BioTox/publication/figures/Figure3/source_data/Figure3_primary_6h.csv` |
| Figure 3 | `/data/kyungan/scripts/BioTox/publication/figures/Figure3/source_data/Figure3_paired_24h.csv` |
| Figure 4 | `/data/kyungan/scripts/BioTox/publication/figures/Figure4/source_data/Figure4_pathway_gsea_per_compound.csv` |
| Figure 4 | `/data/kyungan/scripts/BioTox/publication/figures/Figure4/source_data/Figure4_pathway_selection.csv` |
| Figure 4 | `/data/kyungan/scripts/BioTox/publication/figures/Figure4/source_data/Figure4_top_global_coefficients.csv` |
| Figure 5 | `/data/kyungan/scripts/BioTox/publication/figures/Figure5/source_data/Figure5_compound_selection_statistics.csv` |
| Figure 5 | `/data/kyungan/scripts/BioTox/publication/figures/Figure5/source_data/Figure5_heatmap_displayed_compounds.csv` |
| Figure 5 | `/data/kyungan/scripts/BioTox/publication/figures/Figure5/source_data/Figure5_heatmap_cells.csv` |
| Figure 5 alias | `/data/kyungan/scripts/BioTox/publication/figures/Figure5/source_data/Figure5_all_compound_rankings.csv` |
| Figure S2 | `/data/kyungan/scripts/BioTox/publication/figures/Figure5/source_data/Figure5_compound_selection_statistics.csv` |
| Figure S2 | `/data/kyungan/scripts/BioTox/publication/figures/FigureS2/source_data/FigureS2_displayed_vertical_structure_cases.csv` |

### Known manuscript typo

Only one manuscript value was found to be inconsistent with the source data:
the Figure S2 Gossypol--Niclosamide example.

| Item | Value |
|---|---:|
| Original source-data difference | `0.400834 -> 0.012816` |
| Manuscript value | `0.041 -> 0.013` |
| Correct rounded value | approximately `0.401 -> 0.013` |

The Figure S2 cases are derived from
`Figure5/source_data/Figure5_compound_selection_statistics.csv`; they are not
an independent statistical analysis. The remaining Figure S2 cases and the
Figure 2--5 statistics checked against the manuscript were classified as
matching.

## Source-data provenance

The paths above are the frozen inputs in this publication bundle. The original
upstream analysis path was recorded for Figure 2 in the historical source
registry at:

`/data/kyungan/scripts/BioTox/backup/publication_pre_release_20260902/figures_full_history/source_data/SOURCE_REGISTRY.csv`

`/data/kyungan/scripts/BioTox/publication/_run_output/chem_offset_valid_tuned_classical/publication_summary/stage1_topk_ensemble_curve_revised.csv`

That registry records the same SHA-256 for the canonical upstream file and the
bundled Figure 2 source file:
`1e46566353426226977e795c92e6511e1f1c047778a0f03fa7f71307eba97961`.
The recorded `_run_output` path is not currently present on the filesystem,
so it is provenance evidence rather than a currently available input path.
The Figure 2 alias also matches:

`/data/kyungan/scripts/BioTox/backup/figures_unadjusted_primary_20260826/Figure2/stage1_topk_global_macro_auprc.csv`

For Figures 3--5 and Figure S2, the current `config.json` and generation
manifests record the frozen files under
`/data/kyungan/scripts/BioTox/publication/figures/*/source_data/`, but do not
record a unique original upstream path. Candidate backup directories must not
be treated as the original source without a matching hash or registry entry.
The authoritative current input declarations are in:

- `Figure2/code/config.json`
- `Figure3/code/config.json`
- `Figure4/code/config.json`
- `Figure5/code/config.json`
- `FigureS2/code/config.json`

## Manuscript table provenance

The table generators and their canonical input paths are defined in:

`/home/kyungan/scripts/BioTox/publication/tables/code/table_paths.json`

| Table set | Original input filepath(s) |
|---|---|
| Main Tables 1--2 and performance supplements | `/data/kyungan/scripts/BioTox/backup/nested_cv_k4_calibrated_lbfgs_repeated20x5_combined_v1_publication_summary_v1/table_calibrated_20x5_detailed.csv` and `/data/kyungan/scripts/BioTox/backup/nested_cv_k4_calibrated_lbfgs_repeated20x5_combined_v1/` |
| Main Table 3 | `/data/kyungan/scripts/BioTox/backup/main_table3_tox21_lincs_config_v1/table3_tox21_lincs_numeric.csv` |
| Supplementary Table S1 | `/data/kyungan/pretrain-gnns/dataset/tox21/raw/tox21_smiles.csv` plus the configured Stage-1 run and cohort inputs |
| Supplementary Table S2 | `/data/kyungan/scripts/BioTox/backup/matched_split_rebuilt/` |
| Supplementary Table S3 | `/data/kyungan/scripts/BioTox/backup/matched_split_rebuilt/` and `/data/kyungan/scripts/BioTox/backup/publication_pre_release_20260902/figures_full_history/source_data/paired_timepoint_inputs_6h24h_8to12uM_v1/RUN_COMPLETE.json` |
| Supplementary Table S4 | `/data/kyungan/scripts/BioTox/backup/publication_pre_release_20260902/figures_full_history/source_data/paired_timepoint_inputs_6h24h_8to12uM_v1/RUN_COMPLETE.json` |
| Supplementary Tables S5--S6 | `/data/kyungan/scripts/BioTox/publication/standalone/chem/run` and the configured primary Stage-1 run |
| Supplementary Table S7 | `/data/kyungan/scripts/BioTox/backup/chemical_ensemble_auprc_table_csv_v1/k4_ensemble_composition.csv` |
| Supplementary Tables S8--S9 | `/data/kyungan/scripts/BioTox/backup/nested_cv_k4_calibrated_lbfgs_repeated20x5_combined_v1_publication_summary_v1/table_calibrated_20x5_detailed.csv` and the configured Stage-2 run directory |

The generated table package is written under:

`/data/kyungan/scripts/BioTox/publication/tables/_generated/`

The supplementary generator writes `table_s0_dataset_flow.csv` through
`table_s7_k4_ensemble_composition.csv` there. The primary generator writes
the standardized AUPRC, residualized AUPRC, AUROC, calibration, and related
forest-plot outputs under its run-specific `tables/`, `figures/`, and `data/`
subdirectories. Existing checked-in files under
`/data/kyungan/scripts/BioTox/publication/tables/Table1/`, `Table2/`, and
`TableS1/`--`TableS9/` are not overwritten by regeneration.

The current table configuration points to the old/uncorrected 20x5 nested-CV
run under `/data/kyungan/scripts/BioTox/backup/`. These are the documented
source paths used for the manuscript comparison; they should not be described
as a corrected-estimator provenance chain until `table_paths.json` is updated.

## Regeneration

Run the wrapper from this directory:

```bash
python regenerate_corrected_figures.py
```

It reads each figure's `code/config.json`, validates the configured input
files, and invokes the canonical `generate_figure.py` entry point. Figure 4's
generator contains the retained publication logic formerly spread across the
versioned `Rebuild_Figure5_per_endpoint_panels_*.py` scripts; those older
versions are preserved under `Figure4/code/deprecated/` and are not active.
