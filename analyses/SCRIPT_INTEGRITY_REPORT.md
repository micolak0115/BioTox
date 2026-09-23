# Analysis script integrity report

- Indexed scripts: 31
- Passed: 31
- Failed: 0
- Unindexed method scripts: none
- Missing indexed scripts: none
- Checks: unique numbering, portable filename, indexed SHA-256, AST parse, and static compile.

| Section | Sequence | Script | Role | Provenance | Status |
|---|---:|---|---|---|---|
| 2.2 | 01 | `data_preparation/m2_01_run_lincs_preprocessing_pipeline.py` | entry point | not bundled | PASS |
| 2.2 | 02 | `data_preparation/m2_02_load_lincs_level3_profiles.py` | pipeline stage | not bundled | PASS |
| 2.2 | 03 | `data_preparation/m2_03_compute_auxiliary_lincs_embeddings.py` | pipeline stage | not bundled | PASS |
| 2.2 | 04 | `data_preparation/m2_04_merge_lincs_batches.py` | pipeline stage | not bundled | PASS |
| 2.2 | 05 | `data_preparation/m2_05_filter_lincs_profiles.py` | pipeline stage | not bundled | PASS |
| 2.2 | 06 | `data_preparation/m2_06_align_to_plate_dmso_controls.py` | pipeline stage | not bundled | PASS |
| 2.2 | 07 | `data_preparation/m2_07_aggregate_lincs_replicates.py` | pipeline stage | not bundled | PASS |
| 2.2 | 08 | `data_preparation/m2_08_join_extended_lincs_profiles.py` | pipeline stage | not bundled | PASS |
| 2.2 | 09 | `data_preparation/m2_09_guard_preprocessing_memory.py` | supporting module | not bundled | PASS |
| 2.2 | 10 | `data_preparation/m2_10_build_matched_cohort_and_scaffold_split.py` | entry point | hash differs (historical) | PASS |
| 2.2 | 11 | `data_preparation/m2_11_build_landmark_gene_features.py` | entry point | hash differs (historical) | PASS |
| 2.2 | 12 | `data_preparation/m2_12_prepare_paired_timepoint_contexts.py` | entry point | hash differs (historical) | PASS |
| 2.3 | 01 | `model_specification/m3_01_fit_endpoint_specific_molecular_candidates.py` | entry point | hash differs (historical) | PASS |
| 2.3 | 02 | `model_specification/m3_02_select_topk_molecular_ensemble.py` | entry point | hash match | PASS |
| 2.3 | 03 | `model_specification/m3_03_deploy_frozen_k4_ensemble.py` | entry point | hash match | PASS |
| 2.3 | 04 | `model_specification/m3_04_optimize_offset_ridge_lbfgs.py` | supporting module | hash match | PASS |
| 2.3 | 05 | `model_specification/m3_05_fit_calibrated_nested_cv_fold.py` | supporting module | hash differs (historical) | PASS |
| 2.3 | 06 | `model_specification/m3_06_repeat_nested_scaffold_cv.py` | supporting module | hash match | PASS |
| 2.3 | 07 | `model_specification/m3_07_run_primary_20x5_analysis.py` | entry point | hash match | PASS |
| 2.3 | 08 | `model_specification/m3_08_run_paired_timepoint_20x5_analysis.py` | entry point | hash match | PASS |
| 2.3 | 09 | `model_specification/m3_09_paired_timepoint_runner_base.py` | supporting module | hash match | PASS |
| 2.3 | 10 | `model_specification/m3_10_summarize_repeated_cv.py` | entry point | hash differs (historical) | PASS |
| 2.3 | 11 | `model_specification/m3_11_nadeau_bengio_corrected_test.py` | supporting module | hash match | PASS |
| 2.3 | 12 | `model_specification/m3_12_repeated_cv_inference.py` | supporting module | hash match | PASS |
| 2.3 | 13 | `model_specification/m3_13_aggregate_offset_statistics.py` | supporting module | hash match | PASS |
| 2.3 | 14 | `model_specification/m3_14_shared_stage2_utilities.py` | supporting module | hash differs (historical) | PASS |
| 2.3 | 15 | `model_specification/m3_15_run_calibration_offset_sensitivity.py` | entry point | hash match | PASS |
| 2.3 | 16 | `model_specification/m3_16_optimize_fixed_offset_ridge.py` | supporting module | hash match | PASS |
| 2.4 | 01 | `posthoc_analysis/m4_01_compute_compound_rescue_summaries.py` | entry point | hash match | PASS |
| 2.4 | 02 | `posthoc_analysis/m4_02_shared_scaffold_and_rescue_utilities.py` | supporting module | hash match | PASS |
| 2.4 | 03 | `posthoc_analysis/m4_03_build_structure_pathway_atlas.py` | entry point | hash match | PASS |
