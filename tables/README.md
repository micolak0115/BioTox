# Final tables

`Table1` and `Table2` are legacy publication-table asset directories. The
checked-in `Table1/Table1_source_data.csv` and `Table1/Table1_display.csv` are
deprecated and must not be used for reproduction or validation; see
`Table1/DEPRECATED.md`. `TableS1`--`TableS9` are the canonical supplementary
table sections. Each directory contains the source CSV used for the
corresponding table.

## Supplementary numbering

- `TableS1`: endpoint-specific chemical-only dataset statistics
- `TableS2`: nuclear-receptor cohort sizes and eligibility
- `TableS3`: stress-response cohort sizes and eligibility
- `TableS4`: classical molecular-model settings
- `TableS5`: deep molecular-model settings
- `TableS6`: repeated outer-loop evaluation
- `TableS7`: inner-loop ridge selection
- `TableS8`: molecular-intercept calibration effect
- `TableS9`: context-specific model performance

Generation entry points are in `code/`; Figure S8 carries the calibration-effect visualization.

## Byte-identical provenance summary

The checked-in Table S8 numeric source is byte-identical to the archived
publication-build output. This verifies the asset copy, but does not imply
that the archived calibration-ablation run is the current standardized
primary reproduction run; its run manifest records the residualized variant.

| Asset | Archived byte-identical source | SHA256 | Verification |
| --- | --- | --- | --- |
| `/home/kyungan/scripts/BioTox/publication/tables/TableS8/TableS8_source_data.csv` | `/data/kyungan/scripts/BioTox/backup/publication_pre_release_20260902/tables_full_history/09_figure3_calibration_ablation_statistics/table_s8_figure3_panel_statistics_numeric.csv` | `9e7e2e8cf58b879fe1a1a89d182174b2ee91eae84c8538b49c91fb1c5c952a10` | byte-identical |

## Regenerating

`code/regenerate_tables.py --output-dir <new-dir>` rebuilds the full table
bundle from canonical inputs listed in `code/table_paths.json`, writing to a
fresh directory (existing `Table1/`, `Table2/`, `TableS1/`-`TableS9/` are
never overwritten). See `code/README.md` for details.
