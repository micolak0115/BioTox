# Stage 1 seed-1 full GPU refit

This directory contains a new, isolated fit of the complete molecular
(Stage 1) candidate library followed by endpoint-specific K=4 ensemble
selection and deployment. Existing manuscript and frozen-reference results
were read only and were not overwritten.

## Run contract

- Fixed random seed: `1`.
- Frozen input cohort: `data/generated/molecular_prediction_and_ensemble_manuscript_reproduction/cohort/`.
- Candidate cache reuse: **none** (`cache_source_dir: null`).
- Actual fits: 12 candidates for each of 12 Tox21 endpoints (144
  endpoint-candidate fits).
- GPUs: device 0 and 1, both NVIDIA GeForce RTX 3090.
- Ensemble: endpoint-specific ranking on the frozen development-selection
  partition, followed by a K=4 arithmetic mean.
- Output safety: all candidate, selection, and deployment outputs were written
  below this new directory. The pipeline refuses to run if any enabled output
  directory already exists.

The configuration used was
`pipeline/configs/molecular_prediction_and_ensemble_seed1_gpu_refit.json`:

```bash
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble_seed1_gpu_refit.json \
  validate
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble_seed1_gpu_refit.json \
  run --dry-run
conda run -n BioTox python pipeline/molecular_prediction_and_ensemble.py \
  --config pipeline/configs/molecular_prediction_and_ensemble_seed1_gpu_refit.json \
  run
```

## Reproducibility result

The run completed, but the numerical reproducibility check against the frozen
reference is **FAIL_NUMERICAL_REPRODUCIBILITY**. This status is intentionally
separate from successful pipeline execution.

- Structural check: **PASS**. The frozen cohort, seed, endpoint set,
  12-candidate library, sample counts, positive-label counts, cohort compound
  IDs, and K=4 deployment procedure match.
- Numerical check: **FAIL**. The refit is not numerically identical to the
  frozen candidate predictions or deployed molecular offsets.
- Top-4 overlap is 2--4 candidates per endpoint.
- Mean K=4 AUPRC difference (refit minus reference) is approximately
  `-0.00769`, with endpoint differences from `-0.06797` to `+0.04155`.
- The frozen cache does not retain Stage-1 test-row identifiers. Its test-label
  order differs from the current deterministic loader even though test size
  and positive count match, so direct elementwise test-prediction comparison
  would be invalid.
- Deployed offsets are aligned by explicit compound ID and still differ;
  therefore the audit does not claim numerical reproduction.

The authoritative report is `REPRODUCIBILITY_CHECK_v2.json`. The earlier
`REPRODUCIBILITY_CHECK.json` is a preserved preliminary structural report and
is superseded by the v2 report; neither file is overwritten by the checker.

```bash
conda run -n BioTox python pipeline/stage1/check_seed1_refit_reproducibility.py \
  --reference-root data/generated/molecular_prediction_and_ensemble_manuscript_reproduction \
  --refit-root data/generated/reproducibility_checks/stage1_seed1_gpu_refit_20260923_v1 \
  --output data/generated/reproducibility_checks/stage1_seed1_gpu_refit_20260923_v1/REPRODUCIBILITY_CHECK_v2.json
```

## Outputs

- `candidates/`: 12 newly fitted task caches plus candidate performance and
  full-candidate-mean deployment tables.
- `topk_selection/`: new candidate rankings, K=4 selections, summaries, and
  diagnostic plots.
- `deployed_k4/`: new K=4 molecular offset, deployment report, and manifest.
- `REPRODUCIBILITY_CHECK_v2.json`: authoritative structural and numerical
  comparison with the frozen reference.
- `_launch_attempt_1/`: preserved empty log and stale PID from an initial
  detached launch that was terminated by the execution sandbox before model
  fitting began. It is not part of the completed run.
