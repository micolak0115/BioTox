# Publication tables

Each active publication table has a table-local source-data generator and
configuration. These scripts consume raw data or already completed Stage-1/
Stage-2 result files; they do not launch model fitting, repeated CV, figure
rendering, or another table's generator.

## Directory structure

```text
tables/
├── README.md
├── tables_config.json                 # Wrapper routing only
├── table_generators.json              # Manuscript label and source provenance
├── regenerate_corrected_tables.py     # Runs selected table-local generators
├── Table1/                            # Static manuscript table; no generator
├── Table2/
└── TableS1/ ... TableS9/
    ├── code/config.json               # Input-file contract
    ├── code/generate_tables.py        # Table-specific source-data generator
    ├── source_data/                   # Checked-in table inputs, when applicable
    └── generation/                    # Regenerated table data and validation
```

`tables_config.json` identifies the script, config, and output directory for
each table. Input paths remain in the corresponding `Table*/code/config.json`.
The wrapper validates those inputs before invoking a generator.

## Regeneration

Run from `publication/` with the `BioTox` environment:

```bash
conda run -n BioTox python tables/regenerate_corrected_tables.py
```

Regenerate only selected tables:

```bash
conda run -n BioTox python tables/regenerate_corrected_tables.py \
  --tables TableS2,TableS3
```

Outputs are isolated in each table's `generation/` directory. Table 1 is
literal LaTeX under manuscript label `tab:tox21-lincs-configurations` and is
intentionally excluded from the active generator list.

Table captions and manuscript labels are documented from
`../manuscript/BioTox_BiB_Submission_20260921.tex`; generator provenance is in
`table_generators.json` and `provenance.json`.
