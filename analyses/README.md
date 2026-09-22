# Publication analysis scripts

This directory contains the method-aligned source snapshots for the analyses
reported in the manuscript. Filenames are lowercase, portable, and ordered by
Methods subsection and execution sequence:

```text
m<section>_<subsection>_<sequence>_<purpose>.py
```

For example, `m2_3_07_run_primary_20x5_analysis.py` is the seventh archived
script associated with Methods Section 2.3.

`METHODS_ANALYSIS_SCRIPT_INDEX.csv` is the authoritative map from each script
to its methodological role, historical in-repository source, checksum, and
purpose. Historical source paths are provenance only; the files in this
directory are the canonical public snapshots. A clone does not need the
git-ignored `backup/` directory to inspect or validate them.

The directly runnable, config-driven workflows are under `../standalone/`.
The method-aligned files here intentionally preserve the code used for the
reported analyses, including original imports and command-line defaults.

Validate names, checksums, and syntax:

```bash
python3 shared_01_check_analysis_integrity.py
```

Regenerate the checksum manifest after an intentional update:

```bash
python3 shared_02_build_source_manifest.py
```

Section numbers match the Methods subsections of
`BioTox_BiB_Submission_20260921.tex`: 2.2 ("Input data and biological
context construction") covers LINCS preprocessing, the matched-cohort and
scaffold split, landmark-gene feature construction, and paired-timepoint
context preparation; 2.3 ("Model specification") covers the Stage-1
molecular ensemble (fit, select, freeze) and the Stage-2 calibrated
fixed-offset residual model with repeated scaffold CV; and 2.4 ("Outputs
and post-hoc analyses") covers compound-, gene-, pathway-, and
structural interpretation. Section 2.1 ("Overview") is conceptual and has
no associated scripts.

Within 2.2, sequence reflects a real dependency order, not just topical
grouping: the raw LINCS ingestion/QC pipeline (01-09) must run before the
matched-cohort/scaffold split (10, which reads the pipeline's aggregate
output), which in turn must run before landmark-gene feature construction
(11, which reads both the aggregate and the cohort split) and paired-
timepoint context preparation (12).

`m2_3_10b_build_6cell_landmark_gene_features.py` and
`m2_3_10c_build_time_resolved_gene_features.py` are deprecated stray files
from the unrelated PertRAG/MolRAG project (a different manuscript) and are
intentionally excluded from `METHODS_ANALYSIS_SCRIPT_INDEX.csv`; their
`m2_3_10*` names predate this directory's 2.1-2.4 relabeling and no longer
correspond to any active section -- `shared_01_check_analysis_integrity.py`
will always list them as unindexed for this reason. Their `_run_output/`
subdirectories are likewise not part of this manuscript's outputs.
