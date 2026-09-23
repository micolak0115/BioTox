# Methodology-audit reconstruction: status and open findings (2026-09-23)

## Context

`publication/figures/regenerate_corrected_figures.py` and the current
`publication/tables/` generators are supposed to be sourced from a "corrected"
Stage-2 estimator run (`publication/pipeline/stage2/methodology_audit/results/`).
That directory did not exist anywhere on this host. This note records what was
reconstructed, what was verified, and what remains an open discrepancy that
needs author attention before submission -- none of it was silently patched.

## What was reconstructed

The exact corrected estimator is implemented in
`publication/pipeline/stage2/nested_cv_offset_logistic_calibrated.py`
(`STAGE2_MODEL_SPEC = "frozen_offset_fixed_calibration_intercept_ridge_biology_v3"`,
matching the string `regenerate_corrected_figures.py` checks for). Its two
driver scripts are `run_repeated_calibrated_pipeline.py` (primary 6 h cohort)
and `run_repeated_calibrated_timepoint.py --pert-time 6|24` (paired cohorts).

All three of the module's required inputs (`COHORT_DIR`, `GENE_FEATURES_DIR`,
`CHEM_OFFSET_PATH`) had been relocated to `/data/kyungan/scripts/BioTox/backup/`
without the code being updated. Two small, non-destructive symlinks were added
(not overwriting anything) so the scripts' own default paths resolve correctly:

- `publication/_run_output/paired_timepoint_inputs_6h24h_8to12uM_v1` ->
  `backup/publication_pre_release_20260902/figures_full_history/source_data/paired_timepoint_inputs_6h24h_8to12uM_v1`
- `publication/pipeline/stage2/_run_output/paired_timepoint_inputs_6h24h_8to12uM_v1` -> same target

`primary_6h_20x5` (12 tasks x 4 cells, standardized variant) was run to
completion into `methodology_audit/results/primary_6h_20x5(_summary/_posthoc)`.
Verified: `stage2_model_spec` in the output is exactly `..._v3`; the frozen,
shared calibration intercept is confirmed both in the manifest
(`calibration_intercept_frozen: true`) and directly in the fitting code
(`null_intercept` and `bio_intercept` are logged from the same variable --
never independently re-optimized). This resolves, for the v3 pipeline, the
intercept-sharing discrepancy that had been flagged for the *archived*
pre-correction code (where `bio_intercept != null_intercept`, mean diff
~0.059, confirmed earlier from that code's own OOF output).

`paired_6h_20x5` and `paired_24h_20x5` were not yet run (see "Next steps").

## Open finding: TableS8 ("molecular-intercept calibration effect") panel_b does not reproduce

`analyses/tableS8_standardized_reproduction_20260922/` (built by a separate
agent session reusing this run) reconstructed TableS8's standardized variant
two ways; neither reproduces the published table's panel_b (chem+bio ridge
arm vs. chem-only calibrated arm):

1. Via `run_stage2_calibration_ridge_ablation.py` (an older, separately
   maintained "v1" ablation reimplementation): panel_b comes out ~0 for
   almost every task/cell, while published values are 0.05-0.15.
2. Via the v3 pipeline's own `p_chem_bio`/`p_chem_only` OOF columns directly
   (no ablation script involved): the *opposite* problem for a specific
   subset of tasks -- SR-MMP, SR-p53, and (partially) NR-Aromatase show a
   large, significant (p<0.01-0.03) positive effect at 6 h that the published
   6 h table does not show.

### Root cause, split into two distinct patterns (verified via `alpha_star_outer`
### selected in every one of the 100 outer folds -- see the two attached
### `*_alpha_counts.csv` snapshots)

- **NR-Aromatase/MCF7**: alpha selection is close to a coin flip across outer
  folds -- 47/100 folds select alpha in [0.32, 1000] (real beta signal),
  53/100 select the grid maximum 1e8 (beta collapses to ~0). This is
  consistent with the one-standard-error rule being applied to a validation
  loss curve that is nearly flat for this task: a small amount of fold-level
  noise flips the selected penalty between "no signal" and "real signal."
  This looks like an **inherent instability of the one-SE selection rule**
  for borderline tasks, not a code bug -- worth flagging to the authors as a
  reproducibility caveat, since re-running the identical pipeline can land on
  either regime depending on exact fold composition / floating-point path.

- **SR-MMP/MCF7**: the opposite -- alpha selection is *stable*, not
  borderline. 97/100 outer folds independently select a small alpha (real
  beta signal) across all 20 repeats. This rules out selection noise as the
  explanation. The most likely remaining explanation is that this
  reconstruction's inputs (`backup/gene_features_consistent_v1` cohort/features)
  are not bit-identical to whatever the original methodology-audit run used
  to produce the published TableS8 -- but the original run's exact input
  snapshot no longer exists anywhere on this host, so this cannot be checked
  further here.

### Recommendation

Do not use either reconstruction to silently overwrite the published TableS8
panel_b numbers. Flag both patterns to the authors: (1) the NR-Aromatase-style
instability as a methods caveat about the one-SE ridge-alpha rule's
sensitivity for borderline tasks, and (2) the SR-MMP/SR-p53-style consistent
mismatch as a data-provenance question (was a different gene-feature build
used for the original 6 h audit run?).

## Next steps (not yet done)

- Run `paired_6h_20x5` and `paired_24h_20x5` the same way, to complete the
  three-analysis `methodology_audit/results/` bundle that
  `regenerate_corrected_figures.py` expects.
- Once complete, re-point the relevant table-local `code/config.json` files
  and use `publication/tables/tables_config.json` with
  `regenerate_corrected_tables.py` for the
  new corrected results (currently it still points at the old/uncorrected
  `backup/nested_cv_k4_calibrated_lbfgs_repeated20x5_combined_v1*` run, by
  explicit prior decision, pending exactly this reconstruction).
- Optionally classify all 12 tasks into the "stable-null" / "borderline"
  / "stable-signal-mismatch" buckets identified above, to know how many
  Table1/TableS8/TableS9 rows are affected before deciding how to present
  this to the authors.
