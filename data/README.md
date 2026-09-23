# Publication data

`data/` contains small redistributable inputs, integrity metadata, and outputs
created by the config-driven pipeline. Large or redistribution-restricted
inputs are referenced by configuration but are not replaced by synthetic data.

## Directory structure and files

```text
data/
├── README.md
├── SHA256SUMS                         # SHA-256 hashes for bundled files
├── lincs/
│   └── compoundinfo_beta.txt          # LINCS compound metadata
├── tox21/
│   └── tox21_smiles.csv               # Tox21 compound structures
├── pathways/
│   ├── Enrichr.GO_Biological_Process_2025.gmt
│   ├── Enrichr.Reactome_Pathways_2024.gmt
│   └── MSigDB.Hallmark.2020.gmt
├── matplotlib-cache/
│   └── fontlist-v390.json              # Fixed plotting-font cache
└── generated/                          # Created by pipeline runs
    ├── molecular_prediction_and_ensemble/
    └── transcriptomic_residual_learning/
```

The generated tree contains cohort files, feature matrices, Stage-1 model
outputs, deployed molecular offsets, repeated-CV results, paired-timepoint
inputs, and calibration-sensitivity results. Inspect every current file with:

```bash
find . -type f ! -path './generated/*' -printf '%P\n' | sort
find generated -type f -printf '%P\n' | sort
```

## Integrity check

```bash
sha256sum --check SHA256SUMS
```

The hash file covers bundled files, not large external inputs or newly created
outputs.

## Relationship to the workflows

Stage 1 (`pipeline/stage1/`) reads the LINCS, Tox21, and molecular-model
inputs and writes:

- `generated/molecular_prediction_and_ensemble/cohort/`
- `generated/molecular_prediction_and_ensemble/candidates/`
- `generated/molecular_prediction_and_ensemble/topk_selection/`
- `generated/molecular_prediction_and_ensemble/deployed_k4/`

Stage 2 (`pipeline/stage2/`) consumes the completed Stage-1 cohort, split, and
`deployed_k4/chem_offset_final.csv`, then writes feature, paired-input,
repeated-CV, and calibration-sensitivity results under
`generated/transcriptomic_residual_learning/`. It does not recreate or refit
the Stage-1 outputs.

Run commands are documented in [`../pipeline/README.md`](../pipeline/README.md).

## External inputs

The default configs also reference the aggregate LINCS H5AD, pretrained GNN
weights, and ChemBERTa model snapshot. Their paths are declared in
`../pipeline/configs/*.json`. Reviewers must obtain these inputs from the
corresponding upstream sources or replace the paths with local copies; the
code validates their existence before running. Checked-in manifests may
record historical paths, but current executable configuration uses the
`pipeline/stage1` and `pipeline/stage2` layout.
