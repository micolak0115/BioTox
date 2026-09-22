## Table S7

Table S7. Inner-loop ridge selection. For each outer training set, three-fold inner cross-validation selects the largest eligible ridge penalty within one standard error of the minimum mean validation BCE. Standardization, calibration, and transcriptomic coefficients are estimated from inner training data only.

| section | setting | value | provenance |
| --- | --- | --- | --- |
| Validation | Inner folds | 3 | Corrected completed-run manifests; methodology_audit/METHODS_ALIGNMENT.md; offset_ridge_fixed.py; nested_cv_offset_logistic_calibrated.py |
| Validation | Inner grouping | Stratified Bemis-Murcko scaffold groups | Corrected completed-run manifests; methodology_audit/METHODS_ALIGNMENT.md; offset_ridge_fixed.py; nested_cv_offset_logistic_calibrated.py |
| Validation | Inner selection metric | Unpenalized mean BCE computed stably from logits on inner-validation folds | Corrected completed-run manifests; methodology_audit/METHODS_ALIGNMENT.md; offset_ridge_fixed.py; nested_cv_offset_logistic_calibrated.py |
| Validation | Ridge selection rule | Select the largest λ anywhere on the grid within one SE of the global minimum mean inner-validation BCE; SE is sample SD across three folds / sqrt(3) | Corrected completed-run manifests; methodology_audit/METHODS_ALIGNMENT.md; offset_ridge_fixed.py; nested_cv_offset_logistic_calibrated.py |
| Optimization | Ridge alpha grid | λ = 10^(-4 + k/2), k=0,…,24 (25 values); called alpha in the executable; searched in descending order | Corrected completed-run manifests; methodology_audit/METHODS_ALIGNMENT.md; offset_ridge_fixed.py; nested_cv_offset_logistic_calibrated.py |
