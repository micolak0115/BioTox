# Source provenance

The standalone runtime was copied from the publication code and adapted only at the packaging/configuration boundary. Filenames that only ever carried an iterative-development version suffix (`_v1`, `_v2`, `_v3`) were renamed to their final, unversioned form; every renamed file's own cross-references were updated to match, and its documented functionality is unchanged.

## Chemical workflow

- Main cohort builder: `chem/run/build_matched_cohort.py` (originally `/data/kyungan/scripts/BioTox/publication/build_matched_cohort.py`)
- Candidate fitting: `chem/run/chem_stage1_per_task_offset.py` (originally `/data/kyungan/scripts/BioTox/publication/chem_stage1_per_task_offset.py`)
- Top-K selection: `chem/run/plot_topk_ensemble_publication.py` (originally `/data/kyungan/scripts/BioTox/publication/plot_topk_ensemble_publication.py`)
- K=4 deployment: `chem/run/deploy_stage1_k4_mean.py` (originally `/data/kyungan/scripts/BioTox/publication/deploy_stage1_k4_mean.py`)
- Chemprop D-MPNN candidate (separate `crisp_chemprop_env` subprocess): `chem/run/chemprop_stage1_candidate.py`
- Molecular runtime modules (`model.py`, `loader.py`, `trainer.py`, `splitter.py`, `utils.py`, `featurizer.py`, `finetune.py`, `benchmark.py`, `classical_baseline.py`, `gnn_matched_benchmark.py`): bundled flat under `chem/run/`, originally top-level modules from `/home/kyungan/scripts/BioTox/chem`

Packaging adaptations replace the fixed project `sys.path` with a runtime registration of the original `chem` and `publication` package names against the bundled `chem/run/` and `bio/run/` directories, and expose the pretrained-GNN root, GNN checkpoint variant, ChemBERTa model, Chemprop interpreter, and bundled Chemprop script through the JSON configuration/environment bridge.

## Biological workflow

- Feature construction: `bio/run/build_gene_features.py` (originally `/data/kyungan/scripts/BioTox/publication/build_gene_features.py`)
- Paired-timepoint preparation: `bio/run/prepare_paired_timepoint_inputs.py` (originally `prepare_paired_timepoint_inputs_v1.py`)
- Primary repeated analysis: `bio/run/run_repeated_calibrated_pipeline.py` (originally `run_repeated_calibrated_pipeline_v1.py`), which in turn calls `bio/run/nested_cv_offset_logistic_calibrated.py` (originally `..._v2.py`) and `bio/run/nested_cv_offset_logistic_calibrated_repeated.py` (originally `..._repeated_v1.py`)
- Paired-timepoint analysis: `bio/run/run_repeated_calibrated_timepoint_innerseed.py` (originally `..._innerseed_v3.py`), which wraps `bio/run/run_repeated_calibrated_timepoint.py` (originally `..._v1.py`)
- Calibration sensitivity: `bio/run/run_stage2_calibration_ridge_ablation.py` (originally `..._v1.py`), which compares `bio/run/offset_ridge_fixed.py` (originally `..._v1.py`) against `bio/run/offset_ridge_lbfgs.py` (unversioned in the source tree) as two alternative fixed-offset ridge solvers
- Summary and inference: `bio/run/summarize_repeated_cv_publication.py` (originally `..._v1.py`), `bio/run/aggregate_offset_logistic.py`, `bio/run/nadeau_bengio_test.py`, `bio/run/repeated_cv_inference.py`
- Supporting scaffold splitting and compound-ID utilities are reused directly from `chem/run/splitter.py` and `chem/run/utils.py` rather than duplicated; `bio/run/` never held its own copy of them.

The numerical algorithms, model definitions, split generation, optimization criteria, inference formulas, and overwrite protections remain in the copied runtime modules. The new launchers translate JSON settings into those existing APIs and command-line arguments.
