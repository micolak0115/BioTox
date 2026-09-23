# Publication analysis scripts

This directory contains the method-aligned source snapshots for the analyses
reported in the manuscript. Filenames are lowercase, portable, and ordered by
Methods subsection and execution sequence. Scripts are grouped by process:

```text
<process-directory>/m<section>_<sequence>_<purpose>.py
```

For example, `model_specification/m3_07_run_primary_20x5_analysis.py` is the
seventh archived script associated with Methods Section 2.3.

The process directories are `data_preparation` (Section 2.2),
`model_specification` (Section 2.3), and `posthoc_analysis` (Section 2.4).

`METHODS_ANALYSIS_SCRIPT_INDEX.csv` is the authoritative map from each script
to its methodological role, historical in-repository source, checksum, and
purpose. Historical source paths are provenance only; the files in this
directory are the canonical public snapshots. A clone does not need the
git-ignored `backup/` directory to inspect or validate them.

The directly runnable, config-driven workflows are under `../pipeline/`:
`pipeline/stage1/` contains molecular modeling and ensemble construction, and
`pipeline/stage2/` contains transcriptomic residual modeling. The method-
aligned files here are provenance snapshots, not an additional pipeline that
must be rerun before figure or table generation.

The `historical_source` column may retain former source-tree names such as
`bio/preprocess/...`. Those values identify historical provenance and are not
current Python imports or runnable paths.

Validate names, checksums, and syntax:

```bash
conda run -n BioTox python analyses/shared_01_check_analysis_integrity.py
```

Regenerate the checksum manifest after an intentional update:

```bash
conda run -n BioTox python analyses/shared_02_build_source_manifest.py
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

The archived scripts use the `m<section>_<sequence>_<purpose>.py` naming
scheme after removing the redundant `2_` prefix from the former `m2_*`
filenames. The integrity checker validates the directory, section, sequence,
checksum, and syntax for every indexed script.
