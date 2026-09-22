# Table S1 — Endpoint-specific Stage 1 molecular-pool statistics

This source-data table reports the endpoint-specific label composition of the
common 1,732-compound Stage 1 molecular pool and its fixed scaffold-disjoint
train, validation, and ensemble-selection partitions.

The `*_n_total` columns count unique compounds and are therefore identical
across endpoints within each partition: 1,732 in the pool, 1,039 in train,
346 in validation, and 347 in ensemble selection. The `*_n_labeled`,
`*_n_positive`, `*_n_negative`, and `*_n_missing` columns are endpoint-specific
because Tox21 labels are sparse. For every endpoint and partition,
`n_labeled = n_positive + n_negative` and `n_total = n_labeled + n_missing`.

Primary source: `backup/publication_methods_tables_v6/all_publication_tables_v1/08_dataset_model_settings/table_s1_tox21_endpoint_statistics.csv`.

Construction references:

- `publication/standalone/chem/run/chem_stage1_per_task_offset.py`
- `publication/analyses/m2_2_10_build_matched_cohort_and_scaffold_split.py`
