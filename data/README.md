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
│   ├── compoundinfo_beta.txt          # LINCS compound metadata
│   └── *.h5ad.tar.gz.part_aa..af     # Split aggregate LINCS archive
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

## Reconstruct the aggregate LINCS H5AD

The tracked files
`lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad.tar.gz.part_aa`
through `part_af` are consecutive pieces of one gzip-compressed tar archive;
they are not independently extractable archives. After cloning the repository,
run the following commands from the repository root:

```bash
git lfs pull --include='data/lincs/*.tar.gz.part_*'

archive='data/lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad.tar.gz'

# Confirm that the complete, correctly ordered six-part set is available.
printf '%s\n' "${archive}".part_*
test "$(find data/lincs -maxdepth 1 -type f \
  -name 'lincs_merge_chemical_filter_select_align_aggregate.h5ad.tar.gz.part_*' \
  | wc -l)" -eq 6

# Build through a temporary name, validate it, and avoid partial final files.
test ! -e "$archive" || {
  echo "Refusing to overwrite existing archive: $archive" >&2
  exit 1
}
cat "${archive}".part_* > "${archive}.tmp"
gzip -t "${archive}.tmp"
mv "${archive}.tmp" "$archive"

# Extract the H5AD where the pipeline configurations expect it.
tar -xzf "$archive" -C data/lincs/
(cd data && grep \
  ' lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad$' \
  SHA256SUMS | sha256sum --check -)
```

Successful verification reports:

```text
lincs/lincs_merge_chemical_filter_select_align_aggregate.h5ad: OK
```

The merge order is determined by the zero-padded suffixes (`part_aa`,
`part_ab`, ..., `part_af`). Do not use `tar` on each part separately. If
`gzip -t` fails, remove only the incomplete `.tmp` file, fetch the LFS parts
again, and repeat the merge. The final `.tar.gz` can be retained for archival
purposes or removed after the extracted H5AD passes its checksum.

## Integrity check

```bash
(cd data && sha256sum --check SHA256SUMS)
```

Run this command from the repository root. The hash file covers bundled inputs
and the reconstructed aggregate LINCS H5AD, but not newly generated pipeline
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
