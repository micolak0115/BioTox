# Publication figures

This directory contains the checked-in figure assets, source data, provenance,
and generator code. Figure captions below are taken from
`../manuscript/BioTox_BiB_Submission_20260921.tex`.

## Directory structure

```text
figures/
├── README.md
├── figures_config.json                 # Wrapper routing and copy rules
├── regenerate_corrected_figures.py     # Figure regeneration wrapper
├── Figure1/                            # Static framework artwork
├── Figure2/                            # Ensemble-size selection
├── Figure3/                            # Context-dependent complementarity
├── Figure4/                            # Genes, pathways, and compounds
├── Figure5/                            # Compound-level rescue/correction
├── FigureS1/                           # Static supplementary artwork
└── FigureS2/                           # Structure-case supplementary figure
```

Every figure directory contains its checked-in rendered assets (`.pdf`,
`.eps`, `.svg`, `.png`, or `.tiff` as available). Generated figures also have
`code/`, `source_data/`, `generation/`, `caption.md`, `methods.md`, and/or
`provenance.json`. List every current file with:

```bash
find . -type f ! -path '*/__pycache__/*' -printf '%P\n' | sort
```

## Figure-specific files

| Directory | Files and purpose |
|---|---|
| `Figure1/` | `Figure1.eps`, `.pdf`, `.svg`, `.tiff`: static framework artwork. It has no active generator or source-data directory in this release. |
| `Figure2/` | `code/config.json`, `code/generate_figure.py`: generator and input/output contract; `source_data/Figure2_topk_ensemble_curve.csv` and `Figure2_ensemble_size_selection.csv`: ensemble curves; `generation/`: generated plot, caption, macro-AUPRC CSV, and `RUN_COMPLETE.json`; `Figure2.*`, `caption.md`, `provenance.json`: checked-in assets and provenance. |
| `Figure3/` | `code/config.json`, `code/generate_figure.py`: generator; `source_data/Figure3_all_context_inference.csv`, `Figure3_primary_6h.csv`, `Figure3_paired_24h.csv`: frozen statistics; `generation/`: caption and manifest; `Figure3.*`, `caption.md`, `provenance.json`: assets and provenance. |
| `Figure4/` | `code/config.json`, `code/generate_figure.py`: generator; `source_data/Figure4_pathway_gsea_per_compound.csv`, `Figure4_pathway_selection.csv`, `Figure4_top_global_coefficients.csv`: frozen pathway/gene inputs; `Figure4.*`, `provenance.json`: assets and provenance. |
| `Figure5/` | `code/config.json`, `code/generate_figure5.py`: generator; `source_data/Figure5_all_compound_rankings.csv`, `Figure5_compound_selection_statistics.csv`, `Figure5_heatmap_displayed_compounds.csv`, `Figure5_heatmap_cells.csv`: frozen inputs; `generation/`: layout/global-maxima audits and manifest; `Figure5.*`, `caption.md`, `methods.md`, `provenance.json`: assets, methods, and provenance. |
| `FigureS1/` | `FigureS1.eps`, `.pdf`, `.svg`, `.tiff`: static supplementary artwork. No active generator is included. |
| `FigureS2/` | `code/config.json`, `code/generate_figureS2.py`: generator; `source_data/FigureS2_displayed_vertical_structure_cases.csv`: selected cases; `generation/`: layout audit and manifest; `FigureS2.*`, `caption.md`, `methods.md`, `provenance.json`: assets, methods, and provenance. |

## Regeneration commands

Run from `publication/` after creating `BioTox`:

```bash
conda run -n BioTox python figures/regenerate_corrected_figures.py
```

Regenerate a subset without touching the other figures:

```bash
conda run -n BioTox python figures/regenerate_corrected_figures.py \
  --figures Figure2,Figure3
```

The wrapper reads each figure's `code/config.json`, validates every configured
source file, invokes that figure's generator, copies canonical filenames, and
writes the configured generation manifest. Figure1 and FigureS1 are static and
are not included in the wrapper's active figure list.

Figure generators consume checked-in source data and completed Stage-1/Stage-2
results. They do not fit molecular models, rerun repeated CV, or modify
`data/generated/`.

The wrapper uses the `python` executable from the active environment specified
by the command above; no machine-specific Conda prefix is embedded in the
figure routing configuration.

## Manuscript captions

### Figure 1 — `fig:framework-overview`

**Two-stage framework for context-dependent transcriptomic complementarity.**
Abbreviations: DMSO, dimethyl sulfoxide; trt, treatment. **ALT TEXT:** Flow
diagram illustrating molecular structures and Tox21 assays entering a frozen
molecular-prediction stage, and the fixed predictions themselves with LINCS
transcriptomic profiles together entering a context-matched residual-modeling
stage, and outputs for context-, gene-, and compound-level analyses.

### Figure 2 — `fig:ensemble-size-selection`

**Endpoint-specific molecular ensemble selection.** **(A)** Macro-AUPRC versus
ensemble size (K) for all endpoints and NR and SR subsets. **(B,C)** Endpoint
AUPRC trajectories for NR and SR, respectively. Ensemble constituents are
endpoint specific and define the frozen molecular reference. Abbreviations: NR,
nuclear receptor; SR, stress response; AUPRC, area under the precision-recall
curve. **ALT TEXT:** Line graphs showing held-out AUPRC as ensemble size
increases from one to twelve models. Overall, nuclear-receptor, and
stress-response macro-AUPRC values peak near four models, with separate
trajectories shown for each of the twelve endpoints.

### Figure 3 — `fig:biological-context-matching`

**Context-dependent transcriptomic complementarity.** **(A-D)** Performance
comparison plots across NR-Aromatase, SR-ARE, SR-MMP, and SR-p53 endpoints,
respectively. Left: Bar plots displaying mean AUPRC for molecular-only
(hatched) and transcriptome-complemented models (no hatch) across biological
contexts. Red bars mark the highest AUPRC. Right: Bar plots illustrating paired
ΔAUPRC with Nadeau-Bengio-corrected 95% CI. Boxed cells are either exact or
proxy matched cell lines defined in Table 1. BH adjustment spans twelve
endpoints per context and cohort. Abbreviations: NR, nuclear receptor; SR,
stress response; ARE, antioxidant response element; MMP, mitochondrial
membrane potential; AUPRC, area under the precision-recall curve.

### Figure 4 — `fig:gene-atlas`

**Genes, pathways, and compounds associated with transcriptomic
complementarity.** **(A-D)** NR-Aromatase, SR-ARE, SR-MMP, and SR-p53 endpoints.
Contexts are MCF7 at 24 h except SR-ARE (HepG2, 6 h). Left: Horizontal bar
plots showing top-10 genes with highest absolute ridge coefficients. Right: Bar
plots depicting compound-specific pathway enrichment of top-5 with largest
absolute NES for the selected pathway. **(E)** Heatmap showing cross-endpoint
GSEA screening across compounds. Color intensity gives the fraction of
assay-active compounds with significance (q<0.10).

### Figure 5 — `fig:structural-pathway-atlas`

**Compound-level predictive rescue and correction.** **(A)** Scatter plots
showing predicted probabilities of molecular-only and transcriptome-complemented
models across NR-Aromatase, SR-ARE, SR-MMP, and SR-p53. Contexts are MCF7 at
24 h except SR-ARE (HepG2, 6 h). Annotated compounds indicate the rescued and
corrected compounds with highest probability changes. **(B,C)** Cross-endpoint
rescue and correction heatmaps ranked by cumulative selected change.

### Figure S1

No `fig:` label or corresponding `\caption{}` was found for Figure S1 in the
current manuscript source. The directory therefore documents the checked-in
static artwork only.

### Figure S2

No `fig:` label or corresponding `\caption{}` was found for Figure S2 in the
current manuscript source. Its checked-in `caption.md` and generator remain
available as the asset-level documentation.
