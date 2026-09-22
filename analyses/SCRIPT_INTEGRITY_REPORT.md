# Analysis script integrity report

- Indexed scripts: 31
- Passed: 31
- Failed: 0
- Unindexed method scripts: m2_3_10b_build_6cell_landmark_gene_features.py, m2_3_10c_build_time_resolved_gene_features.py
- Missing indexed scripts: none
- Checks: unique numbering, portable filename, indexed SHA-256, AST parse, and static compile.

| Section | Sequence | Script | Role | Provenance | Status |
|---|---:|---|---|---|---|
| 2.2 | 01 | `m2_2_01_run_lincs_preprocessing_pipeline.py` | entry point | not bundled | PASS |
| 2.2 | 02 | `m2_2_02_load_lincs_level3_profiles.py` | pipeline stage | not bundled | PASS |
| 2.2 | 03 | `m2_2_03_compute_auxiliary_lincs_embeddings.py` | pipeline stage | not bundled | PASS |
| 2.2 | 04 | `m2_2_04_merge_lincs_batches.py` | pipeline stage | not bundled | PASS |
| 2.2 | 05 | `m2_2_05_filter_lincs_profiles.py` | pipeline stage | not bundled | PASS |
| 2.2 | 06 | `m2_2_06_align_to_plate_dmso_controls.py` | pipeline stage | not bundled | PASS |
| 2.2 | 07 | `m2_2_07_aggregate_lincs_replicates.py` | pipeline stage | not bundled | PASS |
| 2.2 | 08 | `m2_2_08_join_extended_lincs_profiles.py` | pipeline stage | not bundled | PASS |
| 2.2 | 09 | `m2_2_09_guard_preprocessing_memory.py` | supporting module | not bundled | PASS |
| 2.2 | 10 | `m2_2_10_build_matched_cohort_and_scaffold_split.py` | entry point | hash differs (historical) | PASS |
| 2.2 | 11 | `m2_2_11_build_landmark_gene_features.py` | entry point | hash match | PASS |
| 2.2 | 12 | `m2_2_12_prepare_paired_timepoint_contexts.py` | entry point | hash match | PASS |
| 2.3 | 01 | `m2_3_01_fit_endpoint_specific_molecular_candidates.py` | entry point | hash match | PASS |
| 2.3 | 02 | `m2_3_02_select_topk_molecular_ensemble.py` | entry point | hash match | PASS |
| 2.3 | 03 | `m2_3_03_deploy_frozen_k4_ensemble.py` | entry point | hash match | PASS |
| 2.3 | 04 | `m2_3_04_optimize_offset_ridge_lbfgs.py` | supporting module | hash match | PASS |
| 2.3 | 05 | `m2_3_05_fit_calibrated_nested_cv_fold.py` | supporting module | hash differs (historical) | PASS |
| 2.3 | 06 | `m2_3_06_repeat_nested_scaffold_cv.py` | supporting module | hash match | PASS |
| 2.3 | 07 | `m2_3_07_run_primary_20x5_analysis.py` | entry point | hash match | PASS |
| 2.3 | 08 | `m2_3_08_run_paired_timepoint_20x5_analysis.py` | entry point | hash match | PASS |
| 2.3 | 09 | `m2_3_09_paired_timepoint_runner_base.py` | supporting module | hash match | PASS |
| 2.3 | 10 | `m2_3_10_summarize_repeated_cv.py` | entry point | hash match | PASS |
| 2.3 | 11 | `m2_3_11_nadeau_bengio_corrected_test.py` | supporting module | hash match | PASS |
| 2.3 | 12 | `m2_3_12_repeated_cv_inference.py` | supporting module | hash match | PASS |
| 2.3 | 13 | `m2_3_13_aggregate_offset_statistics.py` | supporting module | hash match | PASS |
| 2.3 | 14 | `m2_3_14_shared_stage2_utilities.py` | supporting module | hash match | PASS |
| 2.3 | 15 | `m2_3_15_run_calibration_offset_sensitivity.py` | entry point | hash match | PASS |
| 2.3 | 16 | `m2_3_16_optimize_fixed_offset_ridge.py` | supporting module | hash match | PASS |
| 2.4 | 01 | `m2_4_01_compute_compound_rescue_summaries.py` | entry point | hash match | PASS |
| 2.4 | 02 | `m2_4_02_shared_scaffold_and_rescue_utilities.py` | supporting module | hash match | PASS |
| 2.4 | 03 | `m2_4_03_build_structure_pathway_atlas.py` | entry point | hash match | PASS |
