# Main Table 2A | Context-focused complementary prediction

| Endpoint | LINCS context | Exact assay-cell match | N | Chem AUPRC | Chem+Bio AUPRC | ΔAUPRC [NB 95% CI] | P | BH q | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR-Aromatase | MCF7, 24 h | Yes | 1012 | 0.411 ± 0.023 | 0.527 ± 0.039 | +0.117 [+0.013, +0.220] | 0.02724 | 0.109 | Nominal only |
| SR-ARE | HepG2, 6 h | No | 451 | 0.454 ± 0.027 | 0.537 ± 0.030 | +0.083 [+0.010, +0.157] | 0.02642 | 0.3171 | Nominal only |
| SR-MMP | MCF7, 24 h | No | 947 | 0.529 ± 0.019 | 0.651 ± 0.035 | +0.123 [+0.042, +0.203] | 0.00325 | 0.0195 | FDR-supported |
| SR-p53 | MCF7, 24 h | No | 1142 | 0.294 ± 0.017 | 0.499 ± 0.052 | +0.206 [+0.081, +0.330] | 0.001447 | 0.01737 | FDR-supported |

Values use the unadjusted standardized transcriptome and the matched chemical comparator from the same cohort, repeat, and outer fold. P values use the two-sided Nadeau–Bengio corrected paired test. BH q values retain the canonical broader family of 12 endpoints within representation, cell, and source analysis; they were not recomputed only over these selected rows. Exact assay-cell identity is a literal metadata match and is distinct from empirical context selection.
