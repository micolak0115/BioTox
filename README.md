# BioTox publication reproducibility bundle

This directory contains the public-facing code, frozen inputs, generated
publication assets, and manuscript source for the BioTox study. The manuscript
is `manuscript/BioTox_BiB_Submission_20260921.tex`.

## Directory structure

```text
publication/
├── README.md                         # This guide
├── manuscript/                       # Submission LaTeX and bibliography
├── environment/                      # Reproducible Conda specifications
├── data/                             # Small inputs, hashes, and generated outputs
├── pipeline/                         # Config-driven Stage-1/Stage-2 workflows
│   ├── molecular_prediction_and_ensemble.py
│   ├── transcriptomic_residual_learning.py
│   ├── stage1/                       # Molecular-model modules
│   ├── stage2/                       # Transcriptomic residual-model modules
│   └── configs/                      # JSON run configurations
├── figures/                          # Figure assets, source data, generators
├── tables/                           # Table assets, source data, generators
└── analyses/                         # Audits and post-hoc analyses
```

## Reconstruct the split LINCS data archive

The aggregate LINCS input is stored as six Git LFS parts under `data/lincs/`
because the complete file is too large to distribute as one repository file.
From the repository root, download the LFS objects and concatenate the parts in
their filename order:

```bash
git lfs pull --include='data/lincs/*.tar.gz.part_*'

archive='data/lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad.tar.gz'
test ! -e "$archive" || {
  echo "Refusing to overwrite existing archive: $archive" >&2
  exit 1
}
cat "${archive}".part_* > "${archive}.tmp"
gzip -t "${archive}.tmp"
mv "${archive}.tmp" "$archive"
tar -xzf "$archive" -C data/lincs/
```

The wildcard expands lexically from `part_aa` through `part_af`; all six parts
must be present. The extraction creates
`data/lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad`, which is
the path used by both pipeline configs. Verify the extracted file before
running either pipeline:

```bash
(cd data && grep \
  ' lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad$' \
  SHA256SUMS | sha256sum --check -)
```

If the merged archive already exists, retain it or move it elsewhere before
running the reconstruction commands; the example intentionally refuses to
overwrite it. See [`data/README.md`](data/README.md) for the data layout.

## Recreate the environments

From this directory:

```bash
conda env create -f environment/BioTox_environment.yml
conda env create -f environment/BioTox_chemprop_environment.yml
```

The environments are `BioTox` and `BioTox_chemprop`. The Chemprop candidate
uses the separate environment because its Python/PyTorch stack is incompatible
with the main publication environment.

## Run the Stage 1 and Stage 2 pipeline

The default configurations write to `data/generated/` and refuse to overwrite
enabled-stage output directories. Validate first, preview with `--dry-run`,
then run:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble.json validate
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble.json run --dry-run
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble.json run
```

Stage 1 builds the matched cohort, fits molecular candidates, selects the
endpoint-specific top-four ensemble, and deploys the frozen molecular offset.
Stage 2 then consumes those completed Stage-1 outputs:

```bash
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning.json validate
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning.json run --dry-run
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning.json run
```

Stage 2 performs transcriptomic feature construction, primary and paired-
timepoint repeated CV, and calibration-sensitivity analysis. These two
launchers are the only workflows in this release that fit publication models.
All current imports and runtime paths use `stage1` and `stage2`; narrow legacy
`chem`/`bio` aliases exist only to load archived pickle/joblib artifacts.

## Regenerate figures and tables

Figures use checked-in figure source data. Table generators consume completed
raw/intermediate/final result files and do not rerun the analysis pipeline.

```bash
conda run -n BioTox python figures/regenerate_corrected_figures.py
conda run -n BioTox python figures/regenerate_corrected_figures.py \
  --figures Figure2,Figure3
conda run -n BioTox python tables/regenerate_corrected_tables.py
conda run -n BioTox python tables/regenerate_corrected_tables.py \
  --tables TableS8,TableS9
```

Each table-local generator reads completed source/result files; it does not
rerun Stage 1 or Stage 2. Wrapper outputs are written to the corresponding
`tables/Table*/generation/` directory. Table 1 is literal manuscript LaTeX
and has no generator.

## Verification

```bash
conda run -n BioTox python -m compileall -q pipeline analyses figures tables
conda run -n BioTox python -m unittest discover \
  -s pipeline/stage2 -p 'test_*.py' -v
conda run -n BioTox python analyses/shared_01_check_analysis_integrity.py
```

Figure and table captions are taken from the manuscript source. Asset-level
provenance is stored beside the outputs in `figures/*/provenance.json`,
`tables/provenance.json`, and `tables/table_generators.json`.
