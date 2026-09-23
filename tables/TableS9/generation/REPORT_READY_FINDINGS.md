# Report-ready findings: context-matched complementary prediction

## Headline result

Transcriptomic measurements supplied selective, context-dependent information beyond the frozen molecular predictor. Across the four planned reporting contexts—NR-Aromatase–MCF7–24 h, SR-ARE–HepG2–6 h, SR-MMP–MCF7–24 h, and SR-p53–MCF7–24 h—the unadjusted standardized Chem+Bio model improved mean AUPRC in all four comparisons (mean absolute ΔAUPRC +0.132; range +0.083 to +0.206). All 4/4 gains were nominally significant by Nadeau–Bengio corrected paired tests. Under the canonical broader BH family, 2/4 remained FDR-supported: SR-MMP–MCF7–24 h and SR-p53–MCF7–24 h.

## Endpoint-level findings

- **SR-p53–MCF7–24 h showed the largest complementary gain:** Chem AUPRC 0.294 versus Chem+Bio 0.499; ΔAUPRC +0.206 (NB 95% CI +0.081 to +0.331), P=0.00149, BH q=0.0178.
- **SR-MMP–MCF7–24 h was also robust:** Chem AUPRC 0.529 versus Chem+Bio 0.651; ΔAUPRC +0.123 (+0.042 to +0.203), P=0.00321, q=0.0192.
- **NR-Aromatase–MCF7–24 h provided the literal assay-cell-matched example:** Chem AUPRC 0.411 versus Chem+Bio 0.526; ΔAUPRC +0.116 (+0.012 to +0.219), P=0.0285, q=0.114. This is nominal evidence, not broader-family FDR support.
- **SR-ARE–HepG2–6 h showed a nominal hepatic-context gain:** Chem AUPRC 0.452 versus Chem+Bio 0.547; ΔAUPRC +0.095 (+0.023 to +0.168), P=0.0104, q=0.120.

## Exact-match interpretation

Literal Tox21 assay-cell matching should not be equated with universal improvement. The refreshed exact-match audit found directional gains in all three literal matches, but only 1/3 were nominally significant and 0/3 survived the broader BH correction. The defensible conclusion is therefore that complementary signal is **context dependent**, with exact matching biologically informative in some settings but not sufficient by itself to guarantee predictive gain.

## Targeted 24 h MCF7 sensitivity analysis

The latest unadjusted standardized 24 h MCF7 results for NR-Aromatase, SR-MMP, and SR-p53 are provided as targeted sensitivity evidence. These q values answer a narrower sensitivity question and should not replace the broader-family q values in the primary table.

## Recommended manuscript wording

“Unadjusted standardized transcriptomic measurements provided selective complementary prediction beyond the frozen molecular model. In four context-focused comparisons, mean AUPRC increased by 0.095–0.206, with nominal corrected evidence in every comparison. The strongest broader-family FDR-supported gains occurred for SR-MMP and SR-p53 in MCF7 at 24 h. NR-Aromatase in MCF7 at 24 h supplied a biologically aligned exact-cell example, but exact assay-cell matching was not a universal predictor of improvement, indicating that the added value of transcriptomics depends on endpoint, cellular context, and exposure.”

## Reporting constraints

- Use “context-focused” or “endpoint–LINCS context” for the four-row performance table. Reserve “exact match” for literal assay-cell identity.
- Distinguish nominal P values from BH-adjusted q values.
- Keep the broader-family q values in the primary table; present the three-context v3 q values as sensitivity evidence.
- Describe the chemical predictor as frozen and the comparison as paired within the same cohort, repeat, and outer fold.
- CEViChE/viability-residualized results are sensitivity/archive evidence and are not part of this primary bundle.
