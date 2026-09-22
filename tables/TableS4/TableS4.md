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
