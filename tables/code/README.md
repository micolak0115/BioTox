# Table generation code

`generate_table.py` builds a legacy primary publication table bundle (Table1/
Table2 discrimination + calibration performance tables and forest-plot
figures). The checked-in `Table1/` assets are deprecated and must not be used
as reproduction inputs; see `publication/tables/Table1/DEPRECATED.md`.
`generate_supplementary_tables.py` builds the dataset and model
specification tables (TableS1-S9 source data).

Both scripts read their input paths from `table_paths.json` (via
`configure_paths()`) instead of hardcoding them, and always write to a fresh
`--output-dir` -- they refuse to run if it already exists. `table_paths.json`
currently points at the old/uncorrected 20x5 nested-CV run (relocated under
`/data/kyungan/scripts/BioTox/backup/`); reconnecting to a corrected-estimator
source later is a config edit, not a code change.

`regenerate_tables.py` is the wrapper that runs both generators in one call,
driven by `tables_config.json` (script + CLI args per table set), mirroring
`publication/figures/regenerate_corrected_figures.py`. It writes a top-level
`REGENERATION_MANIFEST.json` (per-file sha256) into `--output-dir`.

Existing checked-in tables under `publication/tables/Table1/`, `Table2/`,
`TableS1/`-`TableS9/` are never modified by any of these scripts.

Run from anywhere (paths are absolute in the configs):

```bash
python publication/tables/code/regenerate_tables.py --output-dir <new-dir>
# or individually:
python publication/tables/code/generate_table.py --output-dir <new-dir>
python publication/tables/code/generate_supplementary_tables.py --output-dir <new-dir>
```
