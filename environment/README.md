# Reproducible environments

These Conda specifications support the public BioTox Stage 1/Stage 2 pipeline.
They are separate because the optional Stage-1 Chemprop candidate requires a
newer Python/PyTorch stack than the main workflow.

## Files

| File | Purpose |
|---|---|
| `BioTox_environment.yml` | Main Python 3.9 environment for pipeline, figure, table, and audit code. |
| `BioTox_chemprop_environment.yml` | Python 3.11 environment for the optional Chemprop/D-MPNN Stage-1 candidate. |
| `BioTox_legacy_explicit.txt` | Historical explicit export from the former `crisp` environment; provenance only. |
| `BioTox_chemprop_legacy_explicit.txt` | Historical explicit Chemprop export; provenance only. |
| `README.md` | This guide. |

Complete inventory:

```bash
find . -maxdepth 1 -type f -printf '%f\n' | sort
```

## Installation

From `publication/`:

```bash
conda env create -f environment/BioTox_environment.yml
conda env create -f environment/BioTox_chemprop_environment.yml
```

The resulting environment names are `BioTox` and `BioTox_chemprop`. The YAML
files pin the Python minor version and the scientific packages used by the
publication code. Hardware-specific CUDA resolution may vary by host; CPU-only
reviewers may install CPU-compatible PyTorch wheels in the same environments.

## Smoke checks

```bash
conda run -n BioTox python -c \
  "import numpy, pandas, scipy, sklearn, rdkit, torch; print('BioTox OK')"
conda run -n BioTox_chemprop python -c \
  "import chemprop, torch; print('BioTox_chemprop OK')"
```

The main environment runs `pipeline/stage1/`, `pipeline/stage2/`, figure and
table generation, and integrity checks. The Chemprop interpreter is invoked
only when that Stage-1 candidate is enabled; its path is configured in
`../pipeline/configs/molecular_prediction_and_ensemble.json`.
