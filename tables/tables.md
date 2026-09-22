## Table 1

Biological context correspondence between Tox21 assays and LINCS transcriptomic profiles. Tox21 endpoints were matched to available LINCS profiles according to cellular correspondence, exposure time, and mechanistic relevance. Cellular correspondence was classified as exact, tissue-of-origin proxy, or unmatched. The rightmost column sums cellular (0–2), temporal (0–1), and mechanistic (0–1) correspondence; each star is one point, and a dash in the score column denotes zero. This qualitative score was assigned before model evaluation and is not an empirical performance measure.

Awaiting verified source data; no replacement numerical table has been generated.

## Table 2

Molecular-model performance across the twelve Tox21 endpoints. Endpoint-specific AUPRC and macro-AUPRC are reported for the nuclear-receptor (NR) and stress-response (SR) groups. The best and second-best values in each column are shown in bold and underlined, respectively. The random-baseline AUPRC equals endpoint activity prevalence in the molecular evaluation set. Molecular predictors are frozen; their performance is unchanged by the revised transcriptomic fitting procedure.

| Model | NR-AR | NR-AR-LBD | NR-AhR | NR-Aromatase | NR-ER | NR-ER-LBD | NR-PPAR-gamma | NR macro | SR-ARE | SR-ATAD5 | SR-HSE | SR-MMP | SR-p53 | SR macro |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GAT | 0.138 | 0.087 | 0.365 | 0.165 | 0.263 | 0.049 | 0.073 | 0.163 | 0.371 | 0.140 | 0.129 | 0.351 | 0.211 | 0.240 |
| MACCS + logistic regression | 0.696 | 0.273 | 0.319 | 0.206 | 0.370 | 0.130 | 0.057 | 0.293 | 0.377 | 0.086 | 0.154 | 0.473 | 0.292 | 0.276 |
| RDKit descriptors + XGBoost | 0.616 | 0.341 | 0.472 | 0.221 | 0.307 | 0.195 | 0.150 | 0.329 | 0.480 | 0.178 | 0.154 | 0.475 | 0.292 | 0.316 |
| RDKit descriptors + logistic regression | 0.706 | 0.339 | 0.445 | 0.251 | 0.238 | 0.194 | 0.193 | 0.338 | 0.470 | 0.177 | 0.144 | 0.552 | 0.286 | 0.326 |
| MACCS + XGBoost | 0.616 | 0.374 | 0.375 | 0.360 | 0.384 | **0.264** | 0.101 | 0.353 | 0.539 | 0.110 | 0.111 | 0.513 | 0.339 | 0.323 |
| Morgan + XGBoost | 0.764 | <u>0.381</u> | 0.424 | <u>0.370</u> | 0.331 | 0.153 | 0.246 | <u>0.381</u> | 0.428 | 0.159 | 0.157 | 0.474 | 0.247 | 0.293 |
| Morgan + logistic regression | 0.775 | 0.231 | 0.491 | 0.252 | 0.309 | 0.150 | 0.234 | 0.349 | 0.450 | 0.205 | 0.226 | 0.537 | 0.327 | 0.349 |
| ChemBERTa | 0.744 | 0.367 | 0.450 | 0.283 | <u>0.415</u> | 0.161 | 0.044 | 0.352 | 0.504 | 0.180 | 0.203 | 0.562 | 0.301 | 0.350 |
| Chemprop | 0.734 | 0.232 | 0.373 | 0.251 | 0.335 | 0.121 | <u>0.308</u> | 0.336 | 0.432 | 0.188 | 0.313 | 0.533 | 0.397 | 0.373 |
| GCN | 0.736 | 0.306 | 0.432 | 0.237 | 0.388 | 0.167 | 0.103 | 0.339 | 0.500 | 0.116 | 0.383 | <u>0.572</u> | 0.390 | 0.392 |
| GraphSAGE | 0.802 | 0.342 | 0.448 | 0.350 | 0.376 | 0.183 | 0.074 | 0.368 | 0.508 | 0.131 | 0.321 | 0.544 | 0.431 | 0.387 |
| GIN | <u>0.818</u> | 0.370 | <u>0.512</u> | 0.309 | 0.386 | 0.166 | 0.072 | 0.376 | <u>0.569</u> | <u>0.212</u> | **0.456** | 0.525 | <u>0.436</u> | <u>0.440</u> |
| Task-specific K=4 arithmetic mean | **0.821** | **0.415** | **0.574** | **0.444** | **0.460** | <u>0.219</u> | **0.322** | **0.465** | **0.618** | **0.341** | <u>0.432</u> | **0.622** | **0.527** | **0.508** |
| Random baseline | 0.038 | 0.040 | 0.159 | 0.074 | 0.140 | 0.046 | 0.045 | 0.077 | 0.279 | 0.058 | 0.089 | 0.200 | 0.136 | 0.152 |

## Table S1

Table S1. Endpoint-specific chemical-only dataset statistics for molecular-model construction. The molecular pool contained 1,732 scaffold-disjoint compounds and was partitioned into training (n = 1,039), validation (n = 346), and held-out ensemble-selection (n = 347) sets. Within each endpoint and subset, Labeled is the number of compounds with a Tox21 activity label; Active and Inactive partition the labeled compounds; Missing is the number without an endpoint label.

| family | endpoint | stage1_pool_n_total | stage1_pool_n_labeled | stage1_pool_n_positive | stage1_pool_n_negative | stage1_pool_n_missing | stage1_train_n_total | stage1_train_n_labeled | stage1_train_n_positive | stage1_train_n_negative | stage1_train_n_missing | stage1_validation_n_total | stage1_validation_n_labeled | stage1_validation_n_positive | stage1_validation_n_negative | stage1_validation_n_missing | stage1_ensemble_selection_n_total | stage1_ensemble_selection_n_labeled | stage1_ensemble_selection_n_positive | stage1_ensemble_selection_n_negative | stage1_ensemble_selection_n_missing |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR | NR-AR | 1732 | 1555 | 107 | 1448 | 177 | 1039 | 927 | 83 | 844 | 112 | 346 | 316 | 12 | 304 | 30 | 347 | 312 | 12 | 300 | 35 |
| NR | NR-AR-LBD | 1732 | 1393 | 92 | 1301 | 339 | 1039 | 838 | 70 | 768 | 201 | 346 | 281 | 11 | 270 | 65 | 347 | 274 | 11 | 263 | 73 |
| NR | NR-AhR | 1732 | 1352 | 217 | 1135 | 380 | 1039 | 809 | 125 | 684 | 230 | 346 | 272 | 49 | 223 | 74 | 347 | 271 | 43 | 228 | 76 |
| NR | NR-Aromatase | 1732 | 1081 | 102 | 979 | 651 | 1039 | 645 | 65 | 580 | 394 | 346 | 232 | 22 | 210 | 114 | 347 | 204 | 15 | 189 | 143 |
| NR | NR-ER | 1732 | 1199 | 194 | 1005 | 533 | 1039 | 726 | 132 | 594 | 313 | 346 | 237 | 29 | 208 | 109 | 347 | 236 | 33 | 203 | 111 |
| NR | NR-ER-LBD | 1732 | 1434 | 90 | 1344 | 298 | 1039 | 866 | 63 | 803 | 173 | 346 | 284 | 14 | 270 | 62 | 347 | 284 | 13 | 271 | 63 |
| NR | NR-PPAR-gamma | 1732 | 1268 | 51 | 1217 | 464 | 1039 | 766 | 29 | 737 | 273 | 346 | 255 | 11 | 244 | 91 | 347 | 247 | 11 | 236 | 100 |
| SR | SR-ARE | 1732 | 1109 | 269 | 840 | 623 | 1039 | 684 | 157 | 527 | 355 | 346 | 210 | 52 | 158 | 136 | 347 | 215 | 60 | 155 | 132 |
| SR | SR-ATAD5 | 1732 | 1478 | 73 | 1405 | 254 | 1039 | 885 | 40 | 845 | 154 | 346 | 298 | 16 | 282 | 48 | 347 | 295 | 17 | 278 | 52 |
| SR | SR-HSE | 1732 | 1280 | 94 | 1186 | 452 | 1039 | 781 | 51 | 730 | 258 | 346 | 240 | 20 | 220 | 106 | 347 | 259 | 23 | 236 | 88 |
| SR | SR-MMP | 1732 | 1155 | 262 | 893 | 577 | 1039 | 695 | 163 | 532 | 344 | 346 | 230 | 53 | 177 | 116 | 347 | 230 | 46 | 184 | 117 |
| SR | SR-p53 | 1732 | 1391 | 136 | 1255 | 341 | 1039 | 836 | 70 | 766 | 203 | 346 | 282 | 29 | 253 | 64 | 347 | 273 | 37 | 236 | 74 |

## Table S2

Table S2. Nuclear-receptor cohort sizes and eligibility. n: labeled compounds, the denominator for active percentages. ✓: confirmatory-eligible; △: exploratory; ×: excluded. Bold active counts are < 20; eligibility is not statistical significance. Counts use the full 6-h cohort and compounds profiled at both times for 24 h. Total profiled (6 h/24 h): HA1E, 1,012/861; HepG2, 695/46; HT29, 1,005/606; MCF7, 1,332/1,316.

| Tox21 endpoint | Cell line | Exposure | n | Active | Active (%) | Symbol |
| --- | --- | --- | --- | --- | --- | --- |
| NR-AR | HA1E | 6 | 968 | 72 | 7.4 | ✓ |
| NR-AR | HepG2 | 6 | 668 | 46 | 6.9 | ✓ |
| NR-AR | HT29 | 6 | 960 | 72 | 7.5 | ✓ |
| NR-AR | MCF7 | 6 | 1259 | 82 | 6.5 | ✓ |
| NR-AR-LBD | HA1E | 6 | 869 | 49 | 5.6 | ✓ |
| NR-AR-LBD | HepG2 | 6 | 598 | 37 | 6.2 | ✓ |
| NR-AR-LBD | HT29 | 6 | 862 | 47 | 5.5 | ✓ |
| NR-AR-LBD | MCF7 | 6 | 1145 | 53 | 4.6 | ✓ |
| NR-AhR | HA1E | 6 | 849 | 102 | 12.0 | ✓ |
| NR-AhR | HepG2 | 6 | 585 | 65 | 11.1 | ✓ |
| NR-AhR | HT29 | 6 | 841 | 95 | 11.3 | ✓ |
| NR-AhR | MCF7 | 6 | 1111 | 112 | 10.1 | ✓ |
| NR-Aromatase | HA1E | 6 | 759 | 63 | 8.3 | ✓ |
| NR-Aromatase | HepG2 | 6 | 512 | 47 | 9.2 | ✓ |
| NR-Aromatase | HT29 | 6 | 758 | 63 | 8.3 | ✓ |
| NR-Aromatase | MCF7 | 6 | 1021 | 76 | 7.4 | ✓ |
| NR-ER | HA1E | 6 | 805 | 132 | 16.4 | ✓ |
| NR-ER | HepG2 | 6 | 540 | 92 | 17.0 | ✓ |
| NR-ER | HT29 | 6 | 799 | 128 | 16.0 | ✓ |
| NR-ER | MCF7 | 6 | 1067 | 155 | 14.5 | ✓ |
| NR-ER-LBD | HA1E | 6 | 907 | 55 | 6.1 | ✓ |
| NR-ER-LBD | HepG2 | 6 | 620 | 39 | 6.3 | ✓ |
| NR-ER-LBD | HT29 | 6 | 902 | 53 | 5.9 | ✓ |
| NR-ER-LBD | MCF7 | 6 | 1194 | 61 | 5.1 | ✓ |
| NR-PPAR-γ | HA1E | 6 | 811 | 34 | 4.2 | ✓ |
| NR-PPAR-γ | HepG2 | 6 | 552 | 19 | 3.4 | △ |
| NR-PPAR-γ | HT29 | 6 | 803 | 29 | 3.6 | ✓ |
| NR-PPAR-γ | MCF7 | 6 | 1073 | 34 | 3.2 | ✓ |
| NR-AR | HA1E | 6 and 24 | 830 | 59 | 7.1 | ✓ |
| NR-AR | HepG2 | 6 and 24 | 45 | 5 | 11.1 | × |
| NR-AR | HT29 | 6 and 24 | 585 | 44 | 7.5 | ✓ |
| NR-AR | MCF7 | 6 and 24 | 1243 | 78 | 6.3 | ✓ |
| NR-AR-LBD | HA1E | 6 and 24 | 747 | 39 | 5.2 | ✓ |
| NR-AR-LBD | HepG2 | 6 and 24 | 31 | 1 | 3.2 | × |
| NR-AR-LBD | HT29 | 6 and 24 | 523 | 19 | 3.6 | △ |
| NR-AR-LBD | MCF7 | 6 and 24 | 1129 | 51 | 4.5 | ✓ |
| NR-AhR | HA1E | 6 and 24 | 727 | 91 | 12.5 | ✓ |
| NR-AhR | HepG2 | 6 and 24 | 36 | 6 | 16.7 | × |
| NR-AhR | HT29 | 6 and 24 | 516 | 63 | 12.2 | ✓ |
| NR-AhR | MCF7 | 6 and 24 | 1099 | 111 | 10.1 | ✓ |
| NR-Aromatase | HA1E | 6 and 24 | 655 | 57 | 8.7 | ✓ |
| NR-Aromatase | HepG2 | 6 and 24 | 26 | 14 | 53.8 | × |
| NR-Aromatase | HT29 | 6 and 24 | 475 | 45 | 9.5 | ✓ |
| NR-Aromatase | MCF7 | 6 and 24 | 1012 | 76 | 7.5 | ✓ |
| NR-ER | HA1E | 6 and 24 | 689 | 112 | 16.3 | ✓ |
| NR-ER | HepG2 | 6 and 24 | 23 | 11 | 47.8 | × |
| NR-ER | HT29 | 6 and 24 | 487 | 76 | 15.6 | ✓ |
| NR-ER | MCF7 | 6 and 24 | 1054 | 154 | 14.6 | ✓ |
| NR-ER-LBD | HA1E | 6 and 24 | 772 | 40 | 5.2 | ✓ |
| NR-ER-LBD | HepG2 | 6 and 24 | 39 | 8 | 20.5 | × |
| NR-ER-LBD | HT29 | 6 and 24 | 547 | 25 | 4.6 | ✓ |
| NR-ER-LBD | MCF7 | 6 and 24 | 1178 | 59 | 5.0 | ✓ |
| NR-PPAR-γ | HA1E | 6 and 24 | 696 | 29 | 4.2 | ✓ |
| NR-PPAR-γ | HepG2 | 6 and 24 | 29 | 2 | 6.9 | × |
| NR-PPAR-γ | HT29 | 6 and 24 | 483 | 17 | 3.5 | △ |
| NR-PPAR-γ | MCF7 | 6 and 24 | 1060 | 33 | 3.1 | ✓ |

## Table S3

Table S3. Stress-response cohort sizes and eligibility. n: labeled compounds, the denominator for active percentages. ✓: confirmatory-eligible; △: exploratory; ×: excluded. Bold active counts are < 20; eligibility is not statistical significance. Counts use the full 6-h cohort and compounds profiled at both times for 24 h. Total profiled (6 h/24 h): HA1E, 1,012/861; HepG2, 695/46; HT29, 1,005/606; MCF7, 1,332/1,316.

| Tox21 endpoint | Cell line | Exposure | n | Active | Active (%) | Symbol |
| --- | --- | --- | --- | --- | --- | --- |
| SR-ARE | HA1E | 6 | 690 | 136 | 19.7 | ✓ |
| SR-ARE | HepG2 | 6 | 451 | 98 | 21.7 | ✓ |
| SR-ARE | HT29 | 6 | 682 | 129 | 18.9 | ✓ |
| SR-ARE | MCF7 | 6 | 923 | 160 | 17.3 | ✓ |
| SR-ATAD5 | HA1E | 6 | 911 | 50 | 5.5 | ✓ |
| SR-ATAD5 | HepG2 | 6 | 619 | 33 | 5.3 | ✓ |
| SR-ATAD5 | HT29 | 6 | 903 | 50 | 5.5 | ✓ |
| SR-ATAD5 | MCF7 | 6 | 1196 | 57 | 4.8 | ✓ |
| SR-HSE | HA1E | 6 | 814 | 52 | 6.4 | ✓ |
| SR-HSE | HepG2 | 6 | 547 | 30 | 5.5 | ✓ |
| SR-HSE | HT29 | 6 | 806 | 50 | 6.2 | ✓ |
| SR-HSE | MCF7 | 6 | 1077 | 64 | 5.9 | ✓ |
| SR-MMP | HA1E | 6 | 723 | 129 | 17.8 | ✓ |
| SR-MMP | HepG2 | 6 | 495 | 102 | 20.6 | ✓ |
| SR-MMP | HT29 | 6 | 717 | 121 | 16.9 | ✓ |
| SR-MMP | MCF7 | 6 | 958 | 143 | 14.9 | ✓ |
| SR-p53 | HA1E | 6 | 884 | 89 | 10.1 | ✓ |
| SR-p53 | HepG2 | 6 | 603 | 64 | 10.6 | ✓ |
| SR-p53 | HT29 | 6 | 877 | 85 | 9.7 | ✓ |
| SR-p53 | MCF7 | 6 | 1158 | 100 | 8.6 | ✓ |
| SR-ARE | HA1E | 6 and 24 | 584 | 118 | 20.2 | ✓ |
| SR-ARE | HepG2 | 6 and 24 | 26 | 15 | 57.7 | × |
| SR-ARE | HT29 | 6 and 24 | 416 | 80 | 19.2 | ✓ |
| SR-ARE | MCF7 | 6 and 24 | 913 | 159 | 17.4 | ✓ |
| SR-ATAD5 | HA1E | 6 and 24 | 778 | 43 | 5.5 | ✓ |
| SR-ATAD5 | HepG2 | 6 and 24 | 33 | 6 | 18.2 | × |
| SR-ATAD5 | HT29 | 6 and 24 | 544 | 30 | 5.5 | ✓ |
| SR-ATAD5 | MCF7 | 6 and 24 | 1180 | 55 | 4.7 | ✓ |
| SR-HSE | HA1E | 6 and 24 | 697 | 43 | 6.2 | ✓ |
| SR-HSE | HepG2 | 6 and 24 | 36 | 6 | 16.7 | × |
| SR-HSE | HT29 | 6 and 24 | 498 | 31 | 6.2 | ✓ |
| SR-HSE | MCF7 | 6 and 24 | 1065 | 62 | 5.8 | ✓ |
| SR-MMP | HA1E | 6 and 24 | 613 | 111 | 18.1 | ✓ |
| SR-MMP | HepG2 | 6 and 24 | 29 | 19 | 65.5 | × |
| SR-MMP | HT29 | 6 and 24 | 430 | 61 | 14.2 | ✓ |
| SR-MMP | MCF7 | 6 and 24 | 947 | 141 | 14.9 | ✓ |
| SR-p53 | HA1E | 6 and 24 | 757 | 76 | 10.0 | ✓ |
| SR-p53 | HepG2 | 6 and 24 | 38 | 15 | 39.5 | × |
| SR-p53 | HT29 | 6 and 24 | 533 | 57 | 10.7 | ✓ |
| SR-p53 | MCF7 | 6 and 24 | 1142 | 98 | 8.6 | ✓ |

## Table S4

Table S4. Classical molecular-model settings. Fingerprints are evaluated separately per endpoint; braces denote search spaces.

| Representation | Size | Model | Selection / tuning | Class weighting | Settings |
| --- | --- | --- | --- | --- | --- |
| MACCS structural keys | 167.0 | Logistic regression | C in {0.001, 0.01, 0.1, 1, 10, 100}; selected by validation AUPRC | class_weight=balanced | max_iter=2000 |
| Morgan fingerprint | 2048.0 | Logistic regression | radius=2; C in {0.001, 0.01, 0.1, 1, 10, 100}; selected by validation AUPRC | class_weight=balanced | max_iter=2000 |
| RDKit topological fingerprint | 2048.0 | Logistic regression | C in {0.001, 0.01, 0.1, 1, 10, 100}; selected by validation AUPRC | class_weight=balanced | max_iter=2000 |
| MACCS structural keys | 167.0 | XGBoost | max_depth in {3, 4, 6}; learning_rate in {0.05, 0.1}; selected by validation AUPRC | scale_pos_weight=n_negative/n_positive | n_estimators=300; subsample=0.8; colsample_bytree=0.8; early_stopping_rounds=20; eval_metric=aucpr; n_jobs=1 |
| Morgan fingerprint | 2048.0 | XGBoost | radius=2; max_depth in {3, 4, 6}; learning_rate in {0.05, 0.1}; selected by validation AUPRC | scale_pos_weight=n_negative/n_positive | n_estimators=300; subsample=0.8; colsample_bytree=0.8; early_stopping_rounds=20; eval_metric=aucpr; n_jobs=1 |
| RDKit topological fingerprint | 2048.0 | XGBoost | max_depth in {3, 4, 6}; learning_rate in {0.05, 0.1}; selected by validation AUPRC | scale_pos_weight=n_negative/n_positive | n_estimators=300; subsample=0.8; colsample_bytree=0.8; early_stopping_rounds=20; eval_metric=aucpr; n_jobs=1 |

## Table S5

Table S5. Deep molecular-model settings. Pretrained GNNs: GCN, GAT, GIN, GraphSAGE.

| Representation | Size | Model | Selection / tuning | Class weighting | Settings |
| --- | --- | --- | --- | --- | --- |
| Molecular graph | 300.0 | GCN | Initialized from supervised_contextpred checkpoint; end-to-end single-task fine-tuning; validation AUPRC checkpoint selection | BCE pos_weight=n_negative/n_positive | 5 message-passing layers; hidden_dim=300; mean pooling; JK=last; dropout=0.5; batch_size=32; AdamW lr=1e-4; max_epochs=100; patience=20 |
| Molecular graph | 300.0 | GAT | Initialized from supervised_contextpred checkpoint; end-to-end single-task fine-tuning; validation AUPRC checkpoint selection | BCE pos_weight=n_negative/n_positive | 5 message-passing layers; hidden_dim=300; mean pooling; JK=last; dropout=0.5; batch_size=32; AdamW lr=1e-4; max_epochs=100; patience=20 |
| Molecular graph | 300.0 | GIN | Initialized from supervised_contextpred checkpoint; end-to-end single-task fine-tuning; validation AUPRC checkpoint selection | BCE pos_weight=n_negative/n_positive | 5 message-passing layers; hidden_dim=300; mean pooling; JK=last; dropout=0.5; batch_size=32; AdamW lr=1e-4; max_epochs=100; patience=20 |
| Molecular graph | 300.0 | GraphSAGE | Initialized from supervised_contextpred checkpoint; end-to-end single-task fine-tuning; validation AUPRC checkpoint selection | BCE pos_weight=n_negative/n_positive | 5 message-passing layers; hidden_dim=300; mean pooling; JK=last; dropout=0.5; batch_size=32; AdamW lr=1e-4; max_epochs=100; patience=20 |
| Canonical SMILES tokens | — | DeepChem/ChemBERTa-100M-MLM | End-to-end single-task fine-tuning; validation AUPRC checkpoint selection | BCE pos_weight=n_negative/n_positive | max_length=128; batch_size=32; AdamW lr=1e-4; weight_decay=0; linear warmup=6%; gradient clipping=1.0; max_epochs=20; patience=5 |
| Directed molecular graph | 300.0 | Chemprop D-MPNN | Single-task training; early stopping on validation loss | Unweighted binary classification loss | hidden_dim=300; depth=3; mean aggregation; dropout=0; batch_size=32; max_epochs=100; patience=20 |

## Table S6

Table S6. Repeated outer-loop evaluation. Twenty repetitions of five-fold outer cross-validation yield 100 paired held-out evaluations per endpoint-context pair. Molecular-only and transcriptome-complemented predictions are evaluated on the same held-out compounds, with all fitted quantities estimated from the corresponding outer training set only.

| Section | Setting | Value |
| --- | --- | --- |
| Study design | Chemical prior | Endpoint-specific top-4 arithmetic mean |
| Study design | Chemical prior coefficient | Fixed at 1.0 |
| Study design | Stage-2 intercept | Fit an unpenalized offset-only intercept on each inner/outer training partition, then freeze it for β estimation and predictions |
| Study design | Transcriptomic coefficients | β only; minimize mean unweighted BCE + (λ/2)\|\|β\|\|² with the calibrated molecular offset fixed |
| Input | Cell lines | HA1E, HEPG2, HT29, MCF7 |
| Input | Primary exposure | Primary full-cohort 6 h; paired 6-h and 24-h cohorts evaluated separately |
| Input | Dose window | 8-12 uM |
| Input | Genes | 978 L1000 landmark genes |
| Input | Replicate aggregation | Feature-wise median |
| Preprocessing | No-adjustment variant | Unadjusted perturbation expression; feature-wise training mean/SD (ddof=0); replace SD < 1e-8 by 1; apply training scaler to validation/test |
| Validation | Outer repeats | 20 |
| Validation | Outer folds per repeat | 5 |
| Validation | Outer grouping | Bemis-Murcko scaffold |
| Validation | Distinct outer partitions | Required; duplicate signatures forbidden |
| Validation | Minimum class count | At least 2 positives and 2 negatives in outer train and test |
| Validation | Inner folds | 3 |
| Validation | Inner grouping | Stratified Bemis-Murcko scaffold groups |
| Validation | Inner selection metric | Unpenalized mean BCE computed stably from logits on inner-validation folds |
| Validation | Ridge selection rule | Select the largest λ anywhere on the grid within one SE of the global minimum mean inner-validation BCE; SE is sample SD across three folds / sqrt(3) |
| Optimization | Ridge alpha grid | λ = 10^(-4 + k/2), k=0,…,24 (25 values); called alpha in the executable; searched in descending order |
| Optimization | Solver | L-BFGS-B with analytical gradient |
| Optimization | Primary max iterations | 500 |
| Optimization | Cold-retry max iterations | 2000 |
| Optimization | Gradient tolerance | 1e-6 |
| Optimization | Acceptance gradient | 1e-5 |
| Optimization | Function tolerance | 1e-14 |
| Likelihood | Primary objective | Mean unweighted binary cross-entropy (equal compound weights); no class or sample weighting |
| Metric | Primary discrimination metric | AUPRC |
| Metric | Secondary discrimination metric | AUROC |
| Metric | Calibration metrics | Log loss and Brier score |
| Inference | Paired effect | Chem. + Bio. minus intercept-calibrated Chem. on identical held-out compounds |
| Inference | Dependence correction | Nadeau-Bengio corrected resampled paired t-test on 100 outer-fold effects |
| Inference | Multiple testing | Benjamini-Hochberg within variant and cell across 12 endpoints |
| Interpretation | Confirmatory active-count threshold | At least 20 active compounds per endpoint-cell cohort |
| Reproducibility | Stage-1 seed | 1 |
| Reproducibility | Completed unadjusted outer fits | Primary 6 h: 4,800; paired 6 h: 3,600; paired 24 h: 3,600. All requested fits completed; zero failures. These counts exclude the separate residualized sensitivity variant. |

## Table S7

Table S7. Inner-loop ridge selection. For each outer training set, three-fold inner cross-validation selects the largest eligible ridge penalty within one standard error of the minimum mean validation BCE. Standardization, calibration, and transcriptomic coefficients are estimated from inner training data only.

| Section | Setting | Value |
| --- | --- | --- |
| Study design | Chemical prior | Endpoint-specific top-4 arithmetic mean |
| Study design | Chemical prior coefficient | Fixed at 1.0 |
| Study design | Stage-2 intercept | Fit an unpenalized offset-only intercept on each inner/outer training partition, then freeze it for β estimation and predictions |
| Study design | Transcriptomic coefficients | β only; minimize mean unweighted BCE + (λ/2)\|\|β\|\|² with the calibrated molecular offset fixed |
| Input | Cell lines | HA1E, HEPG2, HT29, MCF7 |
| Input | Primary exposure | Primary full-cohort 6 h; paired 6-h and 24-h cohorts evaluated separately |
| Input | Dose window | 8-12 uM |
| Input | Genes | 978 L1000 landmark genes |
| Input | Replicate aggregation | Feature-wise median |
| Preprocessing | No-adjustment variant | Unadjusted perturbation expression; feature-wise training mean/SD (ddof=0); replace SD < 1e-8 by 1; apply training scaler to validation/test |
| Validation | Outer repeats | 20 |
| Validation | Outer folds per repeat | 5 |
| Validation | Outer grouping | Bemis-Murcko scaffold |
| Validation | Distinct outer partitions | Required; duplicate signatures forbidden |
| Validation | Minimum class count | At least 2 positives and 2 negatives in outer train and test |
| Validation | Inner folds | 3 |
| Validation | Inner grouping | Stratified Bemis-Murcko scaffold groups |
| Validation | Inner selection metric | Unpenalized mean BCE computed stably from logits on inner-validation folds |
| Validation | Ridge selection rule | Select the largest λ anywhere on the grid within one SE of the global minimum mean inner-validation BCE; SE is sample SD across three folds / sqrt(3) |
| Optimization | Ridge alpha grid | λ = 10^(-4 + k/2), k=0,…,24 (25 values); called alpha in the executable; searched in descending order |
| Optimization | Solver | L-BFGS-B with analytical gradient |
| Optimization | Primary max iterations | 500 |
| Optimization | Cold-retry max iterations | 2000 |
| Optimization | Gradient tolerance | 1e-6 |
| Optimization | Acceptance gradient | 1e-5 |
| Optimization | Function tolerance | 1e-14 |
| Likelihood | Primary objective | Mean unweighted binary cross-entropy (equal compound weights); no class or sample weighting |
| Metric | Primary discrimination metric | AUPRC |
| Metric | Secondary discrimination metric | AUROC |
| Metric | Calibration metrics | Log loss and Brier score |
| Inference | Paired effect | Chem. + Bio. minus intercept-calibrated Chem. on identical held-out compounds |
| Inference | Dependence correction | Nadeau-Bengio corrected resampled paired t-test on 100 outer-fold effects |
| Inference | Multiple testing | Benjamini-Hochberg within variant and cell across 12 endpoints |
| Interpretation | Confirmatory active-count threshold | At least 20 active compounds per endpoint-cell cohort |
| Reproducibility | Stage-1 seed | 1 |
| Reproducibility | Completed unadjusted outer fits | Primary 6 h: 4,800; paired 6 h: 3,600; paired 24 h: 3,600. All requested fits completed; zero failures. These counts exclude the separate residualized sensitivity variant. |

## Table S8

Table S8. Effect of molecular-intercept calibration. NR-Aromatase, SR-ARE, SR-MMP, and SR-p53 across four primary 6-h cohorts. Differences are calibrated minus uncalibrated predictions on identical held-out compounds. Mean and SD summarize 20 repeat-level averages; Nadeau–Bengio (NB) intervals/tests use 100 paired outer-fold effects. Negative log-loss and positive AUPRC changes indicate improved probabilistic fit and discrimination, respectively. Reported q-values retain BH adjustment across all twelve endpoints within cell line and metric.

| Family | Tox21 endpoint | Cell line | Compounds | Panel A Δlog loss ± SD | Panel A ΔAUPRC (D0−C0) ± SD | Panel A NB 95% CI | Panel A P | Panel A BH q | Panel B ΔAUPRC ± SD | Panel B NB 95% CI | Panel B P | Panel B BH q |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| NR | NR-AR | HA1E | 968 | -0.1717 ± 0.0046 | +0.0000 ± 0.0000 | [-0.2063, -0.1371] | 2.26e-16 | 9.06e-16 | +0.0873 ± 0.0201 | [+0.0121, +0.1624] | 0.0233 | 0.1396 |
| NR | NR-AR | HEPG2 | 668 | -0.1924 ± 0.0033 | +0.0000 ± 0.0000 | [-0.2315, -0.1532] | 3.77e-16 | 1.51e-15 | +0.1083 ± 0.0407 | [-0.0104, +0.2271] | 0.0733 | 0.2674 |
| NR | NR-AR | HT29 | 960 | -0.1742 ± 0.0035 | +0.0000 ± 0.0000 | [-0.2126, -0.1358] | 1.67e-14 | 5.01e-14 | +0.0848 ± 0.0258 | [-0.0064, +0.1761] | 0.068 | 0.2721 |
| NR | NR-AR | MCF7 | 1259 | -0.1813 ± 0.0046 | +0.0000 ± 0.0000 | [-0.2093, -0.1533] | 8.28e-23 | 4.97e-22 | +0.1180 ± 0.0248 | [+0.0254, +0.2106] | 0.013 | 0.0895 |
| NR | NR-AR-LBD | HA1E | 869 | -0.0308 ± 0.0038 | +0.0000 ± 0.0000 | [-0.0533, -0.0082] | 0.0081 | 0.0081 | -0.0050 ± 0.0133 | [-0.0307, +0.0206] | 0.6979 | 0.9372 |
| NR | NR-AR-LBD | HEPG2 | 598 | -0.0357 ± 0.0045 | +0.0000 ± 0.0000 | [-0.0622, -0.0093] | 0.0087 | 0.0087 | +0.0031 ± 0.0104 | [-0.0264, +0.0326] | 0.8351 | 0.9935 |
| NR | NR-AR-LBD | HT29 | 862 | -0.0340 ± 0.0031 | +0.0000 ± 0.0000 | [-0.0563, -0.0116] | 0.0033 | 0.0033 | -0.0101 ± 0.0133 | [-0.0414, +0.0212] | 0.5224 | 0.7836 |
| NR | NR-AR-LBD | MCF7 | 1145 | -0.0355 ± 0.0040 | +0.0000 ± 0.0000 | [-0.0556, -0.0154] | 0.000699 | 0.000699 | +0.0030 ± 0.0152 | [-0.0374, +0.0433] | 0.8833 | 0.9695 |
| NR | NR-AhR | HA1E | 849 | -0.0817 ± 0.0058 | +0.0000 ± 0.0000 | [-0.1155, -0.0479] | 5.65e-06 | 6.78e-06 | +0.0070 ± 0.0072 | [-0.0302, +0.0442] | 0.7114 | 0.9372 |
| NR | NR-AhR | HEPG2 | 585 | -0.0979 ± 0.0060 | +0.0000 ± 0.0000 | [-0.1317, -0.0641] | 1.03e-07 | 1.54e-07 | -0.0150 ± 0.0222 | [-0.0864, +0.0564] | 0.6771 | 0.9935 |
| NR | NR-AhR | HT29 | 841 | -0.0907 ± 0.0040 | +0.0000 ± 0.0000 | [-0.1225, -0.0588] | 1.53e-07 | 2.29e-07 | +0.0269 ± 0.0118 | [-0.0303, +0.0841] | 0.3536 | 0.7071 |
| NR | NR-AhR | MCF7 | 1111 | -0.0920 ± 0.0066 | +0.0000 ± 0.0000 | [-0.1259, -0.0582] | 4.63e-07 | 6.17e-07 | +0.0207 ± 0.0140 | [-0.0271, +0.0684] | 0.3923 | 0.7845 |
| NR | NR-Aromatase | HA1E | 759 | -0.2353 ± 0.0077 | +0.0000 ± 0.0000 | [-0.2858, -0.1849] | 4.73e-15 | 1.42e-14 | +0.0126 ± 0.0394 | [-0.1507, +0.1758] | 0.8788 | 0.9372 |
| NR | NR-Aromatase | HEPG2 | 512 | -0.2234 ± 0.0100 | +0.0000 ± 0.0000 | [-0.2796, -0.1672] | 4.3e-12 | 1.29e-11 | -0.0006 ± 0.0405 | [-0.1370, +0.1359] | 0.9935 | 0.9935 |
| NR | NR-Aromatase | HT29 | 758 | -0.2336 ± 0.0075 | +0.0000 ± 0.0000 | [-0.2885, -0.1787] | 2.7e-13 | 5.41e-13 | +0.0167 ± 0.0390 | [-0.1347, +0.1681] | 0.8275 | 0.8275 |
| NR | NR-Aromatase | MCF7 | 1021 | -0.2372 ± 0.0067 | +0.0000 ± 0.0000 | [-0.2769, -0.1975] | 1.04e-20 | 3.11e-20 | +0.0617 ± 0.0333 | [-0.0688, +0.1922] | 0.3502 | 0.7845 |
| NR | NR-ER | HA1E | 805 | -0.1811 ± 0.0087 | +0.0000 ± 0.0000 | [-0.2247, -0.1375] | 7.1e-13 | 1.42e-12 | +0.0516 ± 0.0239 | [-0.0110, +0.1141] | 0.1052 | 0.3447 |
| NR | NR-ER | HEPG2 | 540 | -0.1855 ± 0.0084 | +0.0000 ± 0.0000 | [-0.2451, -0.1259] | 1.46e-08 | 3.5e-08 | +0.0566 ± 0.0158 | [-0.0076, +0.1207] | 0.0833 | 0.2674 |
| NR | NR-ER | HT29 | 799 | -0.1785 ± 0.0069 | +0.0000 ± 0.0000 | [-0.2198, -0.1372] | 1.34e-13 | 3.22e-13 | +0.0765 ± 0.0121 | [+0.0059, +0.1470] | 0.0339 | 0.2033 |
| NR | NR-ER | MCF7 | 1067 | -0.1919 ± 0.0090 | +0.0000 ± 0.0000 | [-0.2355, -0.1482] | 6.78e-14 | 1.16e-13 | +0.0629 ± 0.0159 | [+0.0046, +0.1212] | 0.0348 | 0.1392 |
| NR | NR-ER-LBD | HA1E | 907 | -0.2225 ± 0.0045 | +0.0000 ± 0.0000 | [-0.2599, -0.1852] | 1.2e-20 | 7.22e-20 | +0.1527 ± 0.0363 | [+0.0215, +0.2840] | 0.023 | 0.1396 |
| NR | NR-ER-LBD | HEPG2 | 620 | -0.2236 ± 0.0047 | +0.0000 ± 0.0000 | [-0.2638, -0.1834] | 6.16e-19 | 3.69e-18 | +0.1266 ± 0.0462 | [-0.0086, +0.2617] | 0.0662 | 0.2674 |
| NR | NR-ER-LBD | HT29 | 902 | -0.2219 ± 0.0049 | +0.0000 ± 0.0000 | [-0.2612, -0.1827] | 2.41e-19 | 2.89e-18 | +0.1547 ± 0.0293 | [+0.0227, +0.2868] | 0.0221 | 0.2033 |
| NR | NR-ER-LBD | MCF7 | 1194 | -0.2369 ± 0.0043 | +0.0000 ± 0.0000 | [-0.2730, -0.2009] | 3.25e-23 | 3.9e-22 | +0.1309 ± 0.0431 | [-0.0039, +0.2657] | 0.0568 | 0.1705 |
| NR | NR-PPAR-gamma | HA1E | 811 | -0.1026 ± 0.0040 | +0.0000 ± 0.0000 | [-0.1290, -0.0763] | 9.47e-12 | 1.62e-11 | +0.0211 ± 0.0289 | [-0.0744, +0.1166] | 0.6624 | 0.9372 |
| NR | NR-PPAR-gamma | HEPG2 | 552 | -0.1099 ± 0.0060 | +0.0000 ± 0.0000 | [-0.1460, -0.0738] | 2.67e-08 | 4.58e-08 | +0.0595 ± 0.0381 | [-0.0532, +0.1721] | 0.2976 | 0.7142 |
| NR | NR-PPAR-gamma | HT29 | 803 | -0.1077 ± 0.0069 | +0.0000 ± 0.0000 | [-0.1401, -0.0754] | 2.02e-09 | 3.47e-09 | +0.0382 ± 0.0202 | [-0.0414, +0.1179] | 0.3433 | 0.7071 |
| NR | NR-PPAR-gamma | MCF7 | 1073 | -0.1184 ± 0.0046 | +0.0000 ± 0.0000 | [-0.1403, -0.0965] | 2.9e-18 | 6.95e-18 | +0.0483 ± 0.0308 | [-0.0831, +0.1797] | 0.4673 | 0.801 |
| SR | SR-ARE | HA1E | 690 | -0.0736 ± 0.0078 | +0.0000 ± 0.0000 | [-0.1122, -0.0350] | 0.000262 | 0.000285 | -0.0154 ± 0.0179 | [-0.0684, +0.0375] | 0.565 | 0.9372 |
| SR | SR-ARE | HEPG2 | 451 | -0.0648 ± 0.0088 | +0.0000 ± 0.0000 | [-0.1055, -0.0240] | 0.0021 | 0.0023 | -0.0299 ± 0.0208 | [-0.0998, +0.0400] | 0.3984 | 0.7968 |
| SR | SR-ARE | HT29 | 682 | -0.0829 ± 0.0058 | +0.0000 ± 0.0000 | [-0.1181, -0.0476] | 9.7e-06 | 1.16e-05 | -0.0170 ± 0.0178 | [-0.0672, +0.0332] | 0.5041 | 0.7836 |
| SR | SR-ARE | MCF7 | 923 | -0.0843 ± 0.0094 | +0.0000 ± 0.0000 | [-0.1238, -0.0449] | 4.89e-05 | 5.34e-05 | +0.0010 ± 0.0190 | [-0.0551, +0.0572] | 0.9713 | 0.9713 |
| SR | SR-ATAD5 | HA1E | 911 | -0.0563 ± 0.0025 | +0.0000 ± 0.0000 | [-0.0759, -0.0367] | 1.3e-07 | 1.95e-07 | -0.0371 ± 0.0325 | [-0.1425, +0.0683] | 0.4862 | 0.9372 |
| SR | SR-ATAD5 | HEPG2 | 619 | -0.0600 ± 0.0029 | +0.0000 ± 0.0000 | [-0.0831, -0.0368] | 1.34e-06 | 1.79e-06 | +0.0029 ± 0.0379 | [-0.1032, +0.1090] | 0.9571 | 0.9935 |
| SR | SR-ATAD5 | HT29 | 903 | -0.0554 ± 0.0038 | +0.0000 ± 0.0000 | [-0.0773, -0.0334] | 2.46e-06 | 3.28e-06 | +0.0105 ± 0.0246 | [-0.0773, +0.0983] | 0.8128 | 0.8275 |
| SR | SR-ATAD5 | MCF7 | 1196 | -0.0639 ± 0.0038 | +0.0000 ± 0.0000 | [-0.0851, -0.0428] | 3.28e-08 | 4.91e-08 | +0.0125 ± 0.0262 | [-0.0541, +0.0791] | 0.7112 | 0.9695 |
| SR | SR-HSE | HA1E | 814 | -0.2527 ± 0.0050 | +0.0000 ± 0.0000 | [-0.2937, -0.2117] | 1.6900000000000001e-21 | 2.03e-20 | +0.0880 ± 0.0374 | [-0.0218, +0.1977] | 0.1149 | 0.3447 |
| SR | SR-HSE | HEPG2 | 547 | -0.2694 ± 0.0081 | +0.0000 ± 0.0000 | [-0.3162, -0.2226] | 9.3e-20 | 1.12e-18 | +0.1170 ± 0.0358 | [-0.0182, +0.2522] | 0.0891 | 0.2674 |
| SR | SR-HSE | HT29 | 806 | -0.2482 ± 0.0088 | +0.0000 ± 0.0000 | [-0.2985, -0.1979] | 3.18e-16 | 1.91e-15 | +0.1009 ± 0.0222 | [-0.0178, +0.2196] | 0.0947 | 0.2841 |
| SR | SR-HSE | MCF7 | 1077 | -0.2545 ± 0.0075 | +0.0000 ± 0.0000 | [-0.2960, -0.2130] | 2.2500000000000002e-21 | 8.99e-21 | +0.1298 ± 0.0306 | [+0.0258, +0.2337] | 0.0149 | 0.0895 |
| SR | SR-MMP | HA1E | 723 | -0.0820 ± 0.0063 | +0.0000 ± 0.0000 | [-0.1117, -0.0524] | 3.02e-07 | 4.03e-07 | -0.0021 ± 0.0176 | [-0.0541, +0.0499] | 0.9372 | 0.9372 |
| SR | SR-MMP | HEPG2 | 495 | -0.0577 ± 0.0067 | +0.0000 ± 0.0000 | [-0.0928, -0.0225] | 0.0016 | 0.0019 | -0.0107 ± 0.0178 | [-0.0619, +0.0406] | 0.681 | 0.9935 |
| SR | SR-MMP | HT29 | 717 | -0.0864 ± 0.0074 | +0.0000 ± 0.0000 | [-0.1239, -0.0489] | 1.43e-05 | 1.56e-05 | +0.0135 ± 0.0143 | [-0.0371, +0.0640] | 0.5976 | 0.7968 |
| SR | SR-MMP | MCF7 | 958 | -0.0940 ± 0.0069 | +0.0000 ± 0.0000 | [-0.1302, -0.0578] | 1.3e-06 | 1.56e-06 | -0.0034 ± 0.0169 | [-0.0508, +0.0440] | 0.8866 | 0.9695 |
| SR | SR-p53 | HA1E | 884 | -0.1603 ± 0.0068 | +0.0000 ± 0.0000 | [-0.1973, -0.1232] | 1.28e-13 | 3.07e-13 | -0.0095 ± 0.0312 | [-0.1064, +0.0874] | 0.8462 | 0.9372 |
| SR | SR-p53 | HEPG2 | 603 | -0.1529 ± 0.0079 | +0.0000 ± 0.0000 | [-0.2026, -0.1033] | 1.95e-08 | 3.9e-08 | -0.0042 ± 0.0333 | [-0.0877, +0.0794] | 0.9209 | 0.9935 |
| SR | SR-p53 | HT29 | 877 | -0.1613 ± 0.0059 | +0.0000 ± 0.0000 | [-0.1963, -0.1262] | 8.73e-15 | 3.49e-14 | +0.0087 ± 0.0319 | [-0.0607, +0.0781] | 0.8042 | 0.8275 |
| SR | SR-p53 | MCF7 | 1158 | -0.1748 ± 0.0068 | +0.0000 ± 0.0000 | [-0.2102, -0.1394] | 2.97e-16 | 5.95e-16 | -0.0075 ± 0.0254 | [-0.1131, +0.0981] | 0.8887 | 0.9695 |

## Table S9

Table S9. Context-specific model performance across twelve Tox21 endpoints. Molecular-only and transcriptome-complemented predictions were tested on identical held-out compounds per endpoint, cell, and exposure time. Significance on the improvement in AUPRC for transcriptomic complementation is marked as follows: †q < 0.10, ∗q < 0.05, and ∗∗q < 0.01. Endpoints and their AUPRC values are bolded if significant; header (↑) marks endpoints with at least one such displayed gain. ‡: fewer than 20 active compounds.

| Endpoint | Cell | Time (h) | n | Active | Chem. | Chem. + Bio. |
| --- | --- | --- | --- | --- | --- | --- |
| NR-AR | HA1E | 6 | 968 | 72 | 0.461 | 0.461 |
| NR-AR | HepG2 | 6 | 668 | 46 | 0.537 | 0.537 |
| NR-AR | HT29 | 6 | 960 | 72 | 0.457 | 0.457 |
| NR-AR | MCF7 | 6 | 1259 | 82 | 0.459 | 0.459 |
| NR-AR | HA1E | 24 | 830 | 59 | 0.432 | 0.432 |
| NR-AR | HT29 | 24 | 585 | 44 | 0.385 | 0.385 |
| NR-AR | MCF7 | 24 | 1243 | 78 | 0.460 | 0.460 |
| NR-AR-LBD | HA1E | 6 | 869 | 49 | 0.554 | 0.554 |
| NR-AR-LBD | HepG2 | 6 | 598 | 37 | 0.554 | 0.554 |
| NR-AR-LBD | HT29 | 6 | 862 | 47 | 0.561 | 0.561 |
| NR-AR-LBD | MCF7 | 6 | 1145 | 53 | 0.581 | 0.585 |
| NR-AR-LBD | HA1E | 24 | 747 | 39 | 0.503 | 0.503 |
| NR-AR-LBD | HT29 | 24 | 523 | 19 | 0.510 | 0.510† |
| NR-AR-LBD | MCF7 | 24 | 1129 | 51 | 0.583 | 0.586 |
| NR-AhR | HA1E | 6 | 849 | 102 | 0.526 | 0.526 |
| NR-AhR | HepG2 | 6 | 585 | 65 | 0.477 | 0.482 |
| NR-AhR | HT29 | 6 | 841 | 95 | 0.479 | 0.479 |
| NR-AhR | MCF7 | 6 | 1111 | 112 | 0.473 | 0.472 |
| NR-AhR | HA1E | 24 | 727 | 91 | 0.527 | 0.530 |
| NR-AhR | HT29 | 24 | 516 | 63 | 0.521 | 0.521 |
| NR-AhR | MCF7 | 24 | 1099 | 111 | 0.468 | 0.468 |
| **NR-Aromatase** | HA1E | 6 | 759 | 63 | 0.384 | 0.384 |
| **NR-Aromatase** | HepG2 | 6 | 512 | 47 | 0.346 | 0.356 |
| **NR-Aromatase** | HT29 | 6 | 758 | 63 | 0.372 | 0.378 |
| **NR-Aromatase** | MCF7 | 6 | 1021 | 76 | 0.409 | 0.431 |
| **NR-Aromatase** | HA1E | 24 | 655 | 57 | 0.397 | 0.482 |
| **NR-Aromatase** | HT29 | 24 | 475 | 45 | 0.371 | 0.371 |
| **NR-Aromatase** | MCF7 | 24 | 1012 | 76 | 0.411 | **0.538*** |
| NR-ER | HA1E | 6 | 805 | 132 | 0.430 | 0.430 |
| NR-ER | HepG2 | 6 | 540 | 92 | 0.477 | 0.477 |
| NR-ER | HT29 | 6 | 799 | 128 | 0.419 | 0.419 |
| NR-ER | MCF7 | 6 | 1067 | 155 | 0.402 | 0.402 |
| NR-ER | HA1E | 24 | 689 | 112 | 0.441 | 0.441 |
| NR-ER | HT29 | 24 | 487 | 76 | 0.409 | 0.409 |
| NR-ER | MCF7 | 24 | 1054 | 154 | 0.407 | 0.407 |
| NR-ER-LBD | HA1E | 6 | 907 | 55 | 0.385 | 0.384 |
| NR-ER-LBD | HepG2 | 6 | 620 | 39 | 0.404 | 0.404 |
| NR-ER-LBD | HT29 | 6 | 902 | 53 | 0.397 | 0.398 |
| NR-ER-LBD | MCF7 | 6 | 1194 | 61 | 0.379 | 0.418 |
| NR-ER-LBD | HA1E | 24 | 772 | 40 | 0.411 | 0.400 |
| NR-ER-LBD | HT29 | 24 | 547 | 25 | 0.375 | 0.369 |
| NR-ER-LBD | MCF7 | 24 | 1178 | 59 | 0.397 | 0.429 |
| NR-PPAR-gamma | HA1E | 6 | 811 | 34 | 0.235 | 0.235 |
| NR-PPAR-gamma | HepG2 | 6 | 552 | 19 | 0.300 | 0.299† |
| NR-PPAR-gamma | HT29 | 6 | 803 | 29 | 0.193 | 0.192 |
| NR-PPAR-gamma | MCF7 | 6 | 1073 | 34 | 0.222 | 0.222 |
| NR-PPAR-gamma | HA1E | 24 | 696 | 29 | 0.241 | 0.241 |
| NR-PPAR-gamma | HT29 | 24 | 483 | 17 | 0.278 | 0.277† |
| NR-PPAR-gamma | MCF7 | 24 | 1060 | 33 | 0.212 | 0.212 |
| **SR-ARE** | HA1E | 6 | 690 | 136 | 0.474 | 0.479 |
| **SR-ARE** | HepG2 | 6 | 451 | 98 | 0.452 | **0.553*** |
| **SR-ARE** | HT29 | 6 | 682 | 129 | 0.451 | 0.452 |
| **SR-ARE** | MCF7 | 6 | 923 | 160 | 0.452 | 0.459 |
| **SR-ARE** | HA1E | 24 | 584 | 118 | 0.469 | 0.482 |
| **SR-ARE** | HT29 | 24 | 416 | 80 | 0.498 | 0.492 |
| **SR-ARE** | MCF7 | 24 | 913 | 159 | 0.449 | 0.463 |
| SR-ATAD5 | HA1E | 6 | 911 | 50 | 0.295 | 0.293 |
| SR-ATAD5 | HepG2 | 6 | 619 | 33 | 0.372 | 0.368 |
| SR-ATAD5 | HT29 | 6 | 903 | 50 | 0.307 | 0.316 |
| SR-ATAD5 | MCF7 | 6 | 1196 | 57 | 0.254 | 0.260 |
| SR-ATAD5 | HA1E | 24 | 778 | 43 | 0.311 | 0.337 |
| SR-ATAD5 | HT29 | 24 | 544 | 30 | 0.308 | 0.308 |
| SR-ATAD5 | MCF7 | 24 | 1180 | 55 | 0.257 | 0.319 |
| SR-HSE | HA1E | 6 | 814 | 52 | 0.234 | 0.248 |
| SR-HSE | HepG2 | 6 | 547 | 30 | 0.269 | 0.252 |
| SR-HSE | HT29 | 6 | 806 | 50 | 0.254 | 0.251 |
| SR-HSE | MCF7 | 6 | 1077 | 64 | 0.248 | 0.245 |
| SR-HSE | HA1E | 24 | 697 | 43 | 0.231 | 0.238 |
| SR-HSE | HT29 | 24 | 498 | 31 | 0.297 | 0.299 |
| SR-HSE | MCF7 | 24 | 1065 | 62 | 0.239 | 0.254 |
| **SR-MMP** | HA1E | 6 | 723 | 129 | 0.588 | **0.646*** |
| **SR-MMP** | HepG2 | 6 | 495 | 102 | 0.616 | **0.686*** |
| **SR-MMP** | HT29 | 6 | 717 | 121 | 0.563 | **0.643*** |
| **SR-MMP** | MCF7 | 6 | 958 | 143 | 0.527 | **0.635*** |
| **SR-MMP** | HA1E | 24 | 613 | 111 | 0.571 | **0.681*** |
| **SR-MMP** | HT29 | 24 | 430 | 61 | 0.547 | 0.553 |
| **SR-MMP** | MCF7 | 24 | 947 | 141 | 0.529 | **0.659*** |
| **SR-p53** | HA1E | 6 | 884 | 89 | 0.361 | 0.404 |
| **SR-p53** | HepG2 | 6 | 603 | 64 | 0.344 | 0.395 |
| **SR-p53** | HT29 | 6 | 877 | 85 | 0.338 | 0.427 |
| **SR-p53** | MCF7 | 6 | 1158 | 100 | 0.295 | **0.397*** |
| **SR-p53** | HA1E | 24 | 757 | 76 | 0.350 | 0.381 |
| **SR-p53** | HT29 | 24 | 533 | 57 | 0.374 | 0.399 |
| **SR-p53** | MCF7 | 24 | 1142 | 98 | 0.294 | **0.508*** |
