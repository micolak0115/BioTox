# Standalone BioTox analysis workflows

This directory contains reusable, config-driven copies of the publication's two main analytical workflows. The launchers do not use the `Methods2.x` archive names and do not require the original `publication` or `chem` source directories on `PYTHONPATH`; the required Python modules are bundled under each workflow's `run/` directory.

## Layout

```text
standalone/
├── chem/
│   ├── molecular_prediction_and_ensemble.py
│   ├── configs/molecular_prediction_and_ensemble.json
│   └── run/
└── bio/
    ├── transcriptomic_residual_learning.py
    ├── configs/transcriptomic_residual_learning.json
    └── run/
```

Every module under `chem/run/` and `bio/run/` still imports its siblings under their original package names (`chem.xxx`, `publication.xxx`) from before this bundle existed. Each launcher registers those two names against the `run/` directories that actually hold the matching modules today (`publication.utils` resolves to `bio/run/utils.py`, `publication.build_matched_cohort` falls through to `chem/run/build_matched_cohort.py`, and so on) before importing or launching anything, so none of the bundled workflow files needed their internal imports rewritten. `bio/run` therefore still reaches into `chem/run` for shared scaffold-splitting and utility code, exactly as the original codebase did.

All relative paths in a JSON config are resolved relative to that config file, not the current shell directory. Enabled stages refuse to overwrite existing output directories. Generated chemical and biological results are written under `../data/generated/` so inputs and reproducible run products remain together in the publication archive.

The default configurations use immutable inputs vendored in the adjacent
`../data/` directory. See `../data/README.md` and `../data/SHA256SUMS` for the
asset inventory, provenance, and integrity hashes. The ChemBERTa setting may
be either a Hugging Face model identifier or a local model directory; the
default points to the vendored snapshot.

## Environments

Run the launchers in `crisp_env`. The Chemprop candidate is executed by the separate interpreter configured in `inputs.chemprop_python`, normally `crisp_chemprop_env`.

Conda environments are execution dependencies and are not copied into the
data archive. Their explicit package specifications are stored under
`../data/environments/` so they can be recreated independently.

```bash
cd /home/kyungan/scripts/BioTox/publication/standalone
```

## Molecular prediction and ensemble construction

Validate inputs and output paths:

```bash
conda run -n crisp_env python chem/molecular_prediction_and_ensemble.py \
  --config chem/configs/molecular_prediction_and_ensemble.json validate
```

Preview the configured workflow without fitting models:

```bash
conda run -n crisp_env python chem/molecular_prediction_and_ensemble.py \
  --config chem/configs/molecular_prediction_and_ensemble.json run --dry-run
```

Run it:

```bash
conda run -n crisp_env python chem/molecular_prediction_and_ensemble.py \
  --config chem/configs/molecular_prediction_and_ensemble.json run
```

The `steps` list controls the enabled stages. Execution always follows this dependency order:

1. `build_cohort`: match Tox21 and LINCS by connectivity-block InChIKey, exclude LINCS-overlapping scaffolds, and create the molecular development split.
2. `fit_candidates`: fit the twelve endpoint-specific molecular candidates.
3. `select_ensemble`: rank held-out predictions and evaluate top-K arithmetic means.
4. `deploy_ensemble`: deploy the frozen endpoint-specific K=4 ensemble to the matched cohort.

Important chemical settings:

| JSON section | Settings |
|---|---|
| `inputs` | LINCS H5AD, compound metadata, Tox21 structures, pretrained GNN root/variant, ChemBERTa model, and Chemprop interpreter |
| `outputs` | Cohort, candidate cache, top-K selection, and deployed ensemble directories |
| `cohort` | Cells, exposure time, dose window, and scaffold-split fractions |
| `candidate_training` | Endpoint subset, fixed seed, GPU IDs, process count, and optional read-only cache source |
| `ensemble` | Required `k=4`, bootstrap count, and plot DPI |

`candidate_training.n_jobs=1` runs candidates sequentially and avoids joblib process creation. Increase it only when CPU RAM and GPU memory are sufficient.

## Transcriptomic residual learning

The default bio config builds 6-hour landmark-gene features and runs the primary 20 x 5 residual analysis. It expects the cohort and deployed molecular offset produced by the chemical workflow under `../data/generated/molecular_prediction_and_ensemble/`.

Validate:

```bash
conda run -n crisp_env python bio/transcriptomic_residual_learning.py \
  --config bio/configs/transcriptomic_residual_learning.json validate
```

Preview:

```bash
conda run -n crisp_env python bio/transcriptomic_residual_learning.py \
  --config bio/configs/transcriptomic_residual_learning.json run --dry-run
```

Run:

```bash
conda run -n crisp_env python bio/transcriptomic_residual_learning.py \
  --config bio/configs/transcriptomic_residual_learning.json run
```

Available bio stages are:

1. `build_features`: build raw landmark-gene and fixed CEVIChE-projected representations.
2. `prepare_paired_timepoints`: construct matched 6-hour and 24-hour inputs with shared compounds and scaffolds.
3. `run_primary_residual_cv`: run primary repeated nested scaffold CV.
4. `run_paired_6h_residual_cv`: run the paired-cohort 6-hour analysis.
5. `run_paired_24h_residual_cv`: run the paired-cohort 24-hour analysis.
6. `run_calibration_sensitivity`: compare transcriptomic ridge models using calibrated and uncalibrated molecular offsets.

To enable paired-timepoint and calibration analyses, extend `steps`, for example:

```json
"steps": [
  "build_features",
  "prepare_paired_timepoints",
  "run_primary_residual_cv",
  "run_paired_6h_residual_cv",
  "run_paired_24h_residual_cv",
  "run_calibration_sensitivity"
]
```

Important bio settings:

| JSON section | Settings |
|---|---|
| `inputs` | LINCS H5AD, metadata, CEVIChE coefficients, matched cohort, Stage 1 split, and frozen chemical offset |
| `outputs` | Feature, paired-input, repeated-CV, and calibration-sensitivity directories |
| `feature_construction` | Cells, time, dose window, and replicate aggregation |
| `residual_learning` | Representation, endpoints, cells, 20 outer seeds, ridge solver, regularization grid, parallelism, and optimizer tolerances |
| `paired_timepoints` | Paired times and eligible cell lines; HepG2 is excluded from the default paired analysis |
| `calibration_sensitivity` | Source run, representation, endpoint/cell subset, repeat count, and process count |

`residual_learning.n_jobs=1` is the memory-conservative default and avoids unnecessary parallel worker processes. The 25-value ridge grid is represented by `points_per_decade=2.0` over the bundled workflow's fixed range from `1e-4` to `1e8`.

## Partial and resumed runs

Each canonical stage refuses to overwrite its output. To reuse completed upstream stages:

1. Remove those stages from `steps`.
2. Point the corresponding input/output path to the completed directory.
3. Use a new, nonexistent directory for every enabled downstream stage.

For molecular candidate reuse, set `candidate_training.cache_source_dir` to a prior candidate output. The new run still writes to a fresh `outputs.stage1_dir` and validates cached compound order and fixed seed.

## Package validation

Run the standalone package checker after moving or editing the archive:

```bash
conda run -n crisp_env python check_standalone.py
```

The checker parses every Python file, compiles every script, validates both JSON configurations, checks the required runtime files, and rejects hardcoded imports of the original BioTox source tree.
