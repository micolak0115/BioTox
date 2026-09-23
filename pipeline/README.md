# BioTox analysis pipeline

`pipeline/` is the active, config-driven implementation of the publication
analysis. It is self-contained under this directory and has two launchers: one
for molecular Stage 1 and one for transcriptomic Stage 2. Current code imports
the packages as `stage1` and `stage2`; the former directory names are not
runtime entry points.

## Directory structure

```text
pipeline/
├── README.md
├── PROVENANCE.md
├── configs/
│   ├── molecular_prediction_and_ensemble.json
│   ├── molecular_prediction_and_ensemble_manuscript_reproduction.json
│   ├── molecular_prediction_and_ensemble_seed1_reproducibility_check.json
│   ├── molecular_prediction_and_ensemble_seed1_reproducibility_resume.json
│   ├── molecular_prediction_and_ensemble_seed1_gpu_refit.json
│   ├── transcriptomic_residual_learning.json
│   └── transcriptomic_residual_learning_manuscript_reproduction.json
├── molecular_prediction_and_ensemble.py
├── transcriptomic_residual_learning.py
├── stage1/
│   ├── __init__.py
│   ├── benchmark.py
│   ├── build_matched_cohort.py
│   ├── chem_stage1_per_task_offset.py
│   ├── chemprop_stage1_candidate.py
│   ├── check_seed1_reproducibility.py
│   ├── check_seed1_refit_reproducibility.py
│   ├── classical_baseline.py
│   ├── deploy_stage1_k4_mean.py
│   ├── featurizer.py
│   ├── finetune.py
│   ├── gnn_matched_benchmark.py
│   ├── loader.py
│   ├── model.py
│   ├── plot_topk_ensemble_publication.py
│   ├── splitter.py
│   ├── trainer.py
│   └── utils.py
└── stage2/
    ├── __init__.py
    ├── aggregate_offset_logistic.py
    ├── build_gene_features.py
    ├── methods_residual_posthoc.py
    ├── nadeau_bengio_test.py
    ├── nested_cv_offset_logistic_calibrated.py
    ├── nested_cv_offset_logistic_calibrated_repeated.py
    ├── offset_ridge_fixed.py
    ├── offset_ridge_lbfgs.py
    ├── prepare_paired_timepoint_inputs.py
    ├── repeated_cv_inference.py
    ├── run_repeated_calibrated_pipeline.py
    ├── run_repeated_calibrated.py        # unified Stage-2 entry point
    ├── run_repeated_calibrated_timepoint.py
    ├── run_repeated_calibrated_timepoint_innerseed.py
    ├── run_stage2_calibration_ridge_ablation.py
    ├── summarize_repeated_cv_publication.py
    ├── test_methods_posthoc.py
    ├── test_methods_residual_learning.py
    └── utils.py
```

`stage2/methodology_audit/results/` is a stored audit/result tree, not an
additional launcher. It contains completed manifests, split registries,
outer-fold summaries, OOF predictions, alpha diagnostics, beta coefficients,
and post-hoc CSVs. The directory is intentionally large because repeated
20-by-5 analyses produce one file per endpoint/cell/variant. Inspect every
file with:

```bash
find pipeline -type f ! -path '*/__pycache__/*' -printf '%P\n' | sort
```

## Environment and path rules

Run from `publication/` with the `BioTox` environment:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py ...
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py ...
```

Relative paths in a JSON file are resolved relative to that JSON file, not the
current shell directory. Outputs go to `data/generated/` by default. Every
enabled stage refuses to overwrite an existing output directory. Use a new
output directory or remove a deliberately failed run before retrying.

Archived cohort pickles and joblib objects may contain their former Python
module names. `stage1/__init__.py`, `stage1/splitter.py`, and
`stage2/__init__.py` provide compatibility aliases only while loading those
artifacts. New code must import `stage1` or `stage2` directly.

## Molecular Stage 1

`molecular_prediction_and_ensemble.py` is the launcher. Its four stages are:

| Stage | Implementation | Purpose |
|---|---|---|
| `build_cohort` | `stage1/build_matched_cohort.py` | Match Tox21 and LINCS compounds, exclude overlapping scaffolds, and create development splits. |
| `fit_candidates` | `stage1/classical_baseline.py`, `stage1/gnn_matched_benchmark.py`, `stage1/chemprop_stage1_candidate.py`, and model utilities | Fit the endpoint-specific molecular candidate models. |
| `select_ensemble` | `stage1/benchmark.py`, `stage1/plot_topk_ensemble_publication.py` | Evaluate held-out predictions and select the endpoint-specific K=4 arithmetic ensemble. |
| `deploy_ensemble` | `stage1/deploy_stage1_k4_mean.py`, `stage1/chem_stage1_per_task_offset.py` | Deploy the frozen molecular ensemble and write the molecular offset used by Stage 2. |

Validate and preview the default run:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble.json validate
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble.json run --dry-run
```

Run it after validation:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble.json run
```

For the manuscript-matching frozen Stage-1 ensemble, use the dedicated
selection/deployment configuration. It consumes the archived Stage-1 cache and
does not refit candidates. Its outputs are written below
`data/generated/manuscript_replication_run/`, leaving the archived inputs intact:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble_manuscript_reproduction.json \
  run
```

### Seed-1 reproducibility check

The completed check under
`data/generated/reproducibility_checks/stage1_seed1_20260923/` validates the
fixed seed-1 candidate prediction caches and reruns K=4 selection and
deployment in isolated output directories. It does not refit the deep models:
the frozen candidate caches are read-only inputs. This distinction is recorded
as `model_refit_performed: false` in `REPRODUCIBILITY_CHECK.json`.

Use
`molecular_prediction_and_ensemble_seed1_reproducibility_check.json` for a new
cache-reassembly run after changing its output paths. The retained `resume`
configuration records the selection/deployment-only continuation used for the
completed check; its configured output directories now exist, so the launcher
will refuse to run it again and thereby protect the result from overwrite.

`stage1/check_seed1_reproducibility.py` compares all 12 task caches and the
principal selection/deployment CSV files against the frozen reference. The
exact commands and PASS evidence are documented in the check directory's
`README.md`.

A separate full GPU refit with seed 1 is stored under
`data/generated/reproducibility_checks/stage1_seed1_gpu_refit_20260923_v1/`.
It uses `cache_source_dir: null`, fits all 144 endpoint-candidate combinations,
and then reruns K=4 selection and deployment. Its structural audit passes, but
its authoritative `REPRODUCIBILITY_CHECK_v2.json` reports
`FAIL_NUMERICAL_REPRODUCIBILITY`: the newly fitted predictions and deployed
offsets are not numerically identical to the frozen reference. The run README
records the exact command, comparison scope, and limitations.

## Transcriptomic Stage 2

`transcriptomic_residual_learning.py` runs the following stages:

| Stage | Implementation | Purpose |
|---|---|---|
| `build_features` | `stage2/build_gene_features.py` | Build landmark-gene representations from LINCS profiles. |
| `prepare_paired_timepoints` | `stage2/prepare_paired_timepoint_inputs.py` | Construct matched 6-hour/24-hour cohorts. |
| `run_primary_residual_cv` | `stage2/run_repeated_calibrated.py primary` | Run the primary 20-repeat, 5-fold scaffold-disjoint residual analysis. |
| `run_paired_6h_residual_cv` | `stage2/run_repeated_calibrated.py paired-innerseed --pert-time 6` | Run paired-cohort 6-hour analysis. |
| `run_paired_24h_residual_cv` | `stage2/run_repeated_calibrated.py paired-innerseed --pert-time 24` | Run paired-cohort 24-hour analysis. |
| `run_calibration_sensitivity` | `stage2/run_stage2_calibration_ridge_ablation.py` | Compare calibrated and uncalibrated molecular offsets. |

Validate, preview, and run:

```bash
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning.json validate
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning.json run --dry-run
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning.json run
```

For a manuscript replication run from the frozen molecular offset and the
archived Stage-2 feature inputs, use the dedicated configuration below. It
re-runs the primary, paired 6-hour, paired 24-hour, and calibration-sensitivity
analyses without rebuilding input features or overwriting archived results:

```bash
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning_manuscript_reproduction.json \
  validate
conda run -n BioTox python pipeline/transcriptomic_residual_learning.py \
  --config pipeline/configs/transcriptomic_residual_learning_manuscript_reproduction.json \
  run --dry-run
```

The same Stage-2 runners can be invoked directly through the unified entry
point. The mode-specific options are forwarded to the original implementation
without changing their defaults or validation:

```bash
conda run -n BioTox python pipeline/stage2/run_repeated_calibrated.py primary \
  --out-root <primary-output>
conda run -n BioTox python pipeline/stage2/run_repeated_calibrated.py paired-innerseed \
  --pert-time 6 --input-root <paired-input-root> --out-root <paired-6h-output>
conda run -n BioTox python pipeline/stage2/run_repeated_calibrated.py paired-innerseed \
  --pert-time 24 --input-root <paired-input-root> --out-root <paired-24h-output>
```

The similarly named files are intentionally retained as compatibility and
library modules. `nested_cv_offset_logistic_calibrated.py` implements one
outer-fold analysis; `nested_cv_offset_logistic_calibrated_repeated.py`
reuses it across repeated scaffold partitions; `offset_ridge_fixed.py` and
`offset_ridge_lbfgs.py` provide alternative solver implementations; and
`repeated_cv_inference.py` provides statistical helpers. They are therefore
composable or alternative components, not duplicate pipeline entry points.

The default Stage-2 config expects the completed Stage-1 cohort, split, and
`deployed_k4/chem_offset_final.csv` under `data/generated/`. Stage 2 does not
refit or recreate Stage-1 models.

## Supporting modules and tests

- `stage1/featurizer.py`, `loader.py`, `model.py`, `trainer.py`, and `utils.py`
  provide molecular data loading, representations, models, training, and
  shared utilities.
- `stage2/offset_ridge_fixed.py` and `offset_ridge_lbfgs.py` implement the fixed
  offset and ridge fitting variants.
- `stage2/aggregate_offset_logistic.py`, `repeated_cv_inference.py`, and
  `summarize_repeated_cv_publication.py` aggregate and audit completed runs.
- `stage2/methods_residual_posthoc.py`, `nadeau_bengio_test.py`, and `utils.py`
  provide post-hoc statistics and validation utilities.
- `stage2/test_methods_posthoc.py` and `stage2/test_methods_residual_learning.py`
  are focused `unittest` modules.

```bash
conda run -n BioTox python -m unittest discover \
  -s pipeline/stage2 -p 'test_*.py' -v
```
