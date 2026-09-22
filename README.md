# BioTox publication release

This directory is the review and reuse bundle for the manuscript in
`manuscript/main.tex`. Final assets are deliberately boring to locate:
`figures/Figure1` through `Figure5` and `FigureS1`, plus `tables/Table1`,
`Table2`, and `TableS1` through `TableS7`. Each directory contains the exact
display asset and its source data/provenance.

## Reproduce or inspect

Run the release checks from the repository root:

```bash
python3 analysis/utils/release_tools/validate_publication.py
python3 publication/analyses/shared_01_check_analysis_integrity.py
python3 publication/standalone/check_standalone.py
```

The standalone workflows under `standalone/` are the supported rerun entry
points. They require external LINCS, Tox21, viability, pretrained-model, and
ChemBERTa inputs described in `data/README.md`; those large inputs are not
silently substituted by the release.

`analyses/` is the frozen Methods-section archive. `analysis/utils/release_tools/` contains
only deterministic packaging and validation utilities. Superseded working
assets remain recoverable under the ignored `backup/publication_pre_release_*`
archive and are not part of the public execution path.

The manuscript submission bundle is `manuscript/manuscript.zip`; run
`unzip -t` after rebuilding it to verify the archive.
