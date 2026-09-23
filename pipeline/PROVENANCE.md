# Source provenance

The standalone runtime was copied from the publication code and adapted only at the packaging/configuration boundary. Filenames that only ever carried an iterative-development version suffix (`_v1`, `_v2`, `_v3`) were renamed to their final, unversioned form; every renamed file's own cross-references were updated to match, and its documented functionality is unchanged.

## Chemical workflow

- Main cohort builder: `stage1/build_matched_cohort.py` (originally `/data/kyungan/scripts/BioTox/publication/build_matched_cohort.py`)
- Candidate fitting: `stage1/chem_stage1_per_task_offset.py` (originally `/data/kyungan/scripts/BioTox/publication/chem_stage1_per_task_offset.py`)
- Top-K selection: `stage1/plot_topk_ensemble_publication.py` (originally `/data/kyungan/scripts/BioTox/publication/plot_topk_ensemble_publication.py`)
- K=4 deployment: `stage1/deploy_stage1_k4_mean.py` (originally `/data/kyungan/scripts/BioTox/publication/deploy_stage1_k4_mean.py`)
- Chemprop D-MPNN candidate (separate `BioTox_chemprop` subprocess): `stage1/chemprop_stage1_candidate.py`
- Molecular runtime modules (`model.py`, `loader.py`, `trainer.py`, `splitter.py`, `utils.py`, `featurizer.py`, `finetune.py`, `benchmark.py`, `classical_baseline.py`, `gnn_matched_benchmark.py`): bundled flat under `stage1/`, originally top-level modules from `/home/kyungan/scripts/BioTox/chem`

Packaging adaptations replace fixed project paths with the bundled `stage1` and `stage2` packages. Narrow `chem` and `bio` module aliases are retained only so archived Python pickle/joblib artifacts can still resolve their original class module names. The pretrained-GNN root, GNN checkpoint variant, ChemBERTa model, Chemprop interpreter, and bundled Chemprop script remain configurable through the JSON configuration/environment bridge.

## Biological workflow

- Feature construction: `stage2/build_gene_features.py` (originally `/data/kyungan/scripts/BioTox/publication/build_gene_features.py`)
- Paired-timepoint preparation: `stage2/prepare_paired_timepoint_inputs.py` (originally `prepare_paired_timepoint_inputs_v1.py`)
- Primary repeated analysis: `stage2/run_repeated_calibrated_pipeline.py` (originally `run_repeated_calibrated_pipeline_v1.py`), which in turn calls `stage2/nested_cv_offset_logistic_calibrated.py` (originally `..._v2.py`) and `stage2/nested_cv_offset_logistic_calibrated_repeated.py` (originally `..._repeated_v1.py`)
- Paired-timepoint analysis: `stage2/run_repeated_calibrated_timepoint_innerseed.py` (originally `..._innerseed_v3.py`), which wraps `stage2/run_repeated_calibrated_timepoint.py` (originally `..._v1.py`)
- Calibration sensitivity: `stage2/run_stage2_calibration_ridge_ablation.py` (originally `..._v1.py`), which compares `stage2/offset_ridge_fixed.py` (originally `..._v1.py`) against `stage2/offset_ridge_lbfgs.py` (unversioned in the source tree) as two alternative fixed-offset ridge solvers
- Summary and inference: `stage2/summarize_repeated_cv_publication.py` (originally `..._v1.py`), `stage2/aggregate_offset_logistic.py`, `stage2/nadeau_bengio_test.py`, `stage2/repeated_cv_inference.py`
- Supporting scaffold splitting and compound-ID utilities are reused directly from `stage1/splitter.py` and `stage1/utils.py` rather than duplicated; `stage2/` never held its own copy of them.

The numerical algorithms, model definitions, split generation, optimization criteria, inference formulas, and overwrite protections remain in the copied runtime modules. The new launchers translate JSON settings into those existing APIs and command-line arguments.
