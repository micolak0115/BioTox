# Stage 1 seed-1 reproducibility check

This directory is an isolated reproducibility run of the frozen molecular
(Stage 1) K=4 ensemble. No existing candidate, selection, deployment, or
manuscript result was overwritten.

## Scope

- Fixed model seed: `1`.
- Candidate source: the completed seed-1 candidate caches in
  `data/generated/molecular_prediction_and_ensemble_manuscript_reproduction/candidates/`.
- Candidate handling: read-only semantic validation and normalization into
  this directory's `candidates/` tree.
- Ensemble handling: rerun endpoint-specific candidate ranking, K=4
  arithmetic-mean selection, and deployment from the copied predictions.
- Deep-model refit: **not performed**. The default execution sandbox did not
  expose the host's NVIDIA device nodes, so this cache-only run does not claim
  to reproduce neural-network optimization from raw inputs. The host GPUs were
  subsequently confirmed operational and a separate full refit was performed
  under `stage1_seed1_gpu_refit_20260923_v1/`.
- Verification: all 12 candidate caches and five principal selection and
  deployment CSV files match the frozen reference semantically or exactly.

The machine-readable result is `REPRODUCIBILITY_CHECK.json`; its status is
`PASS`, `model_refit_performed` is `false`, and
`existing_results_overwritten` is `false`.

## Commands

The full cache-reassembly configuration is retained at
`pipeline/configs/molecular_prediction_and_ensemble_seed1_reproducibility_check.json`.
After candidate reassembly completed, selection and deployment were resumed
without rerunning or overwriting that output:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble_seed1_reproducibility_resume.json \
  validate
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble_seed1_reproducibility_resume.json \
  run

conda run -n BioTox python pipeline/stage1/check_seed1_reproducibility.py \
  --reference-candidates data/generated/molecular_prediction_and_ensemble_manuscript_reproduction/candidates \
  --reference-selection data/generated/molecular_prediction_and_ensemble_manuscript_reproduction/topk_selection \
  --reference-deployment data/generated/molecular_prediction_and_ensemble_manuscript_reproduction/deployed_k4 \
  --reproduced-root data/generated/reproducibility_checks/stage1_seed1_20260923 \
  --output data/generated/reproducibility_checks/stage1_seed1_20260923/REPRODUCIBILITY_CHECK.json
```

Every enabled pipeline stage refuses to run when its configured output
directory already exists. The checker likewise refuses to overwrite an
existing report.

## Outputs

- `candidates/`: normalized copies of seed-1 candidate predictions and their
  performance summaries.
- `topk_selection/`: rerun Top-K curves, K=4 task selections, summaries, and
  diagnostic plots.
- `deployed_k4/`: deployed K=4 molecular offset, deployment report, and run
  manifest.
- `REPRODUCIBILITY_CHECK.json`: comparisons, hashes, and final PASS status.
