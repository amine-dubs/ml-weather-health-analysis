# Scratch Architectures Report

This report combines the full DNN benchmark outputs already saved in the repository with the primitive-layer scratch runs added for the professor request.

## Executive Summary

The project now covers two levels of from-scratch work:

- Architecture-level PyTorch models: `FlexMLP`, `ResidualMLP`, explicit training loops, optimizers, schedulers, and early stopping.
- Primitive-layer scratch models: `MyLinear`, `MyBatchNorm1d`, `ScratchFlexMLP`, and `ScratchResidualMLP` built directly from parameters and tensor math.

The best-performing model families are summarized below, with the full architecture tables archived in CSV form for easy inspection.

## Best Models by Task

| task | metric | best_dnn_model | best_dnn_value | ml_model | ml_value | winner | scratch_best_config | scratch_best_value | preprocessing_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| anomaly_employee | f1 | MLP(256x128,relu,a=0.001,lr=0.001) | 0.742304 | DBSCAN | 0.4694 | DNN | n/a | n/a | Same train/test files, label encoding + StandardScaler. |
| anomaly_heart | balanced_acc | MLP(64,relu,a=0.0001,lr=0.001) | 0.77108 | Elliptic Envelope | 0.664 | DNN | n/a | n/a | Same anomaly framing and reduced-positive-class imbalance. |
| anomaly_wine | f1 | MLP(128x64x32,relu,a=0.0001,lr=0.001) | 0.993737 | Elliptic Envelope | 0.9188 | DNN | n/a | n/a | Same separate-class anomaly setup + StandardScaler. |
| covid_classification | auc_roc | MLP(128x64,relu,a=0.0001,lr=0.01) | 0.900915 | Stacking (LR) | 0.897468 | DNN | n/a | n/a | KNNImputer(k=5) + StandardScaler + stratify split – identical to ML best. |
| energy_forecasting | r2 | MLP(256x128,relu,a=0.0001,lr=0.001) | 0.997694 | Linear Regression | 0.998542 | ML | n/a | n/a | MinMax scaling, chronological split, window=24. |
| heart_classification | roc_auc | MLP(256x128,relu,a=0.0001,lr=0.01) | 0.94161 | StackingClassifier | 0.978387 | ML | scratch_flex_small | 0.942035 | StandardScaler + stratify split – identical to ML best. |
| multi_output_regression | avg_r2 | MLP(128x64x32,relu,a=0.0001,lr=0.001) | 0.466455 | XGBoost_MultiOutput | 0.9282 | ML | n/a | n/a | Joint KNN imputation on X+y – identical to ML. |
| temperature_regression | r2 | MLP(256x128x64,relu,a=0.0001,lr=0.001) | 0.743923 | Stacking | 0.776645 | ML | scratch_temp_residual_128x2 | 0.7347 | Drop 3 cols, NaN-preserving LabelEncode, KNNImputer(5), StandardScaler – identical to ML. |
| weather_classification | auc_score | MLP(128x64,relu,a=0.001,lr=0.001) | 0.640683 | Random Forest_NoNorm | 0.849285 | ML | n/a | n/a | Same 31 engineered features, distribution_sampling imputation, no scaling. |
| wind_forecasting | r2 | MLP(256x128,relu,a=0.0001,lr=0.01) | 0.970601 | Ridge Regression | 0.971364 | ML | scratch_wind_flex_256_128 | 0.970979 | MinMax scaling, chronological split, window=24. |

## Full Architecture Catalog

The table below is the machine-readable catalog used to build the report. It combines the full hyperparameter search tables for all tasks plus the scratch-layer runs already saved in `pytorch_results/`.

### anomaly_employee

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| anomaly_employee | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | f1 | 0.742304 |
| anomaly_employee | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | f1 | 0.741009 |
| anomaly_employee | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | f1 | 0.739726 |
| anomaly_employee | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | f1 | 0.737902 |
| anomaly_employee | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | f1 | 0.736179 |
| anomaly_employee | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | f1 | 0.736034 |
| anomaly_employee | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | f1 | 0.735748 |
| anomaly_employee | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | f1 | 0.735386 |
| anomaly_employee | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | f1 | 0.734961 |
| anomaly_employee | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | f1 | 0.734864 |
| anomaly_employee | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | f1 | 0.734402 |
| anomaly_employee | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | f1 | 0.732896 |
| anomaly_employee | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | f1 | 0.729322 |
| anomaly_employee | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | f1 | 0.728417 |
| anomaly_employee | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | f1 | 0.71557 |

### anomaly_heart

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| anomaly_heart | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | balanced_acc | 0.77108 |
| anomaly_heart | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | balanced_acc | 0.762081 |
| anomaly_heart | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | balanced_acc | 0.726331 |
| anomaly_heart | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | balanced_acc | 0.711415 |
| anomaly_heart | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | balanced_acc | 0.696499 |
| anomaly_heart | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | balanced_acc | 0.651874 |
| anomaly_heart | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | balanced_acc | 0.645957 |
| anomaly_heart | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | balanced_acc | 0.636958 |
| anomaly_heart | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | balanced_acc | 0.556583 |
| anomaly_heart | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | balanced_acc | 0.556583 |
| anomaly_heart | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | balanced_acc | 0.556583 |
| anomaly_heart | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | balanced_acc | 0.5 |
| anomaly_heart | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | balanced_acc | 0.5 |
| anomaly_heart | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | balanced_acc | 0.5 |
| anomaly_heart | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | balanced_acc | 0.5 |

### anomaly_wine

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| anomaly_wine | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | f1 | 0.993737 |
| anomaly_wine | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | f1 | 0.993724 |
| anomaly_wine | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | f1 | 0.991632 |
| anomaly_wine | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | f1 | 0.991632 |
| anomaly_wine | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | f1 | 0.991632 |
| anomaly_wine | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | f1 | 0.990615 |
| anomaly_wine | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | f1 | 0.988554 |
| anomaly_wine | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | f1 | 0.987526 |
| anomaly_wine | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | f1 | 0.983368 |
| anomaly_wine | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | f1 | 0.983299 |
| anomaly_wine | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | f1 | 0.98242 |
| anomaly_wine | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | f1 | 0.980311 |
| anomaly_wine | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | f1 | 0.979339 |
| anomaly_wine | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | f1 | 0.977035 |
| anomaly_wine | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | f1 | 0.977035 |

### covid_classification

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| covid_classification | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | auc_roc | 0.900915 |
| covid_classification | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | auc_roc | 0.900351 |
| covid_classification | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | auc_roc | 0.89253 |
| covid_classification | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | auc_roc | 0.891404 |
| covid_classification | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | auc_roc | 0.889846 |
| covid_classification | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | auc_roc | 0.884047 |
| covid_classification | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | auc_roc | 0.883948 |
| covid_classification | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | auc_roc | 0.883881 |
| covid_classification | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | auc_roc | 0.875663 |
| covid_classification | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | auc_roc | 0.873608 |
| covid_classification | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | auc_roc | 0.873608 |
| covid_classification | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | auc_roc | 0.87321 |
| covid_classification | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | auc_roc | 0.871322 |
| covid_classification | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | auc_roc | 0.866881 |
| covid_classification | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | auc_roc | 0.814455 |

### energy_forecasting

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| energy_forecasting | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | r2 | 0.996241 |
| energy_forecasting | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | r2 | 0.996128 |
| energy_forecasting | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | r2 | 0.995982 |
| energy_forecasting | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | r2 | 0.995716 |
| energy_forecasting | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | r2 | 0.995522 |
| energy_forecasting | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | r2 | 0.99545 |
| energy_forecasting | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | r2 | 0.995355 |
| energy_forecasting | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | r2 | 0.995234 |
| energy_forecasting | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | r2 | 0.995032 |
| energy_forecasting | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | r2 | 0.995013 |
| energy_forecasting | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | r2 | 0.99465 |
| energy_forecasting | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | r2 | 0.994115 |
| energy_forecasting | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | r2 | 0.993923 |
| energy_forecasting | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | r2 | 0.993599 |
| energy_forecasting | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | r2 | 0.992819 |

### heart_classification

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| heart_classification | pytorch-advanced | adv_weighted_bce | val_auc | 0.958251 |
| heart_classification | pytorch-advanced | adv_wider | val_auc | 0.956656 |
| heart_classification | pytorch-advanced | adv_warmup_cosine | val_auc | 0.955446 |
| heart_classification | pytorch-advanced | adv_residual_wide | val_auc | 0.955226 |
| heart_classification | pytorch-advanced | adv_residual | val_auc | 0.954455 |
| heart_classification | pytorch-advanced | adv_best_combo | val_auc | 0.95209 |
| heart_classification | pytorch-standard | pt_sgd_momentum | val_auc | 0.960066 |
| heart_classification | pytorch-standard | pt_baseline | val_auc | 0.957536 |
| heart_classification | pytorch-standard | pt_dropout_04 | val_auc | 0.957316 |
| heart_classification | pytorch-standard | pt_dropout_02 | val_auc | 0.955941 |
| heart_classification | pytorch-standard | pt_full_modern | val_auc | 0.955886 |
| heart_classification | pytorch-standard | pt_scheduler_cosine | val_auc | 0.95462 |
| heart_classification | pytorch-standard | pt_batchnorm | val_auc | 0.953245 |
| heart_classification | pytorch-standard | pt_adamw | val_auc | 0.953135 |
| heart_classification | pytorch-standard | pt_bn_dropout | val_auc | 0.952695 |
| heart_classification | pytorch-standard | pt_scheduler_plateau | val_auc | 0.952585 |
| heart_classification | scratch-layers | scratch_flex_small | val_auc | 0.953685 |
| heart_classification | scratch-layers | scratch_residual_128x2 | val_auc | 0.953465 |
| heart_classification | scratch-layers | scratch_flex_256_128 | val_auc | 0.953355 |
| heart_classification | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | roc_auc | 0.94161 |
| heart_classification | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | roc_auc | 0.928005 |
| heart_classification | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | roc_auc | 0.920351 |
| heart_classification | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | roc_auc | 0.914683 |
| heart_classification | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | roc_auc | 0.912132 |
| heart_classification | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | roc_auc | 0.912132 |
| heart_classification | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | roc_auc | 0.911139 |
| heart_classification | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | roc_auc | 0.906533 |
| heart_classification | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | roc_auc | 0.904974 |
| heart_classification | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | roc_auc | 0.899022 |
| heart_classification | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | roc_auc | 0.8964 |
| heart_classification | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | roc_auc | 0.8964 |
| heart_classification | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | roc_auc | 0.8964 |
| heart_classification | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | roc_auc | 0.891865 |
| heart_classification | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | roc_auc | 0.88981 |

### multi_output_regression

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| multi_output_regression | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | pressure_r2 | 0.216418 |
| multi_output_regression | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | pressure_r2 | 0.209004 |
| multi_output_regression | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | pressure_r2 | 0.206916 |
| multi_output_regression | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | pressure_r2 | 0.203065 |
| multi_output_regression | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | pressure_r2 | 0.201844 |
| multi_output_regression | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | pressure_r2 | 0.201668 |
| multi_output_regression | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | pressure_r2 | 0.19802 |
| multi_output_regression | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | pressure_r2 | 0.186493 |
| multi_output_regression | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | pressure_r2 | 0.178576 |
| multi_output_regression | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | pressure_r2 | 0.176687 |
| multi_output_regression | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | pressure_r2 | 0.161386 |
| multi_output_regression | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | pressure_r2 | 0.154752 |
| multi_output_regression | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | pressure_r2 | -0.000032 |
| multi_output_regression | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | pressure_r2 | -0.000337 |
| multi_output_regression | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | pressure_r2 | -0.001273 |

### temperature_regression

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| temperature_regression | pytorch-advanced | adv_residual_wide | test_r2 | 0.752036 |
| temperature_regression | pytorch-advanced | adv_residual | test_r2 | 0.74929 |
| temperature_regression | pytorch-advanced | adv_best_combo | test_r2 | 0.748703 |
| temperature_regression | pytorch-advanced | adv_wider | test_r2 | 0.74626 |
| temperature_regression | pytorch-advanced | adv_warmup_cosine | test_r2 | 0.744506 |
| temperature_regression | pytorch-advanced | adv_huber | test_r2 | 0.738325 |
| temperature_regression | pytorch-standard | pt_adamw | test_r2 | 0.748625 |
| temperature_regression | pytorch-standard | pt_scheduler_plateau | test_r2 | 0.747974 |
| temperature_regression | pytorch-standard | pt_baseline | test_r2 | 0.747962 |
| temperature_regression | pytorch-standard | pt_batchnorm | test_r2 | 0.747462 |
| temperature_regression | pytorch-standard | pt_scheduler_cosine | test_r2 | 0.746984 |
| temperature_regression | pytorch-standard | pt_sgd_momentum | test_r2 | 0.743962 |
| temperature_regression | pytorch-standard | pt_bn_dropout | test_r2 | 0.742322 |
| temperature_regression | pytorch-standard | pt_full_modern | test_r2 | 0.738491 |
| temperature_regression | pytorch-standard | pt_dropout_02 | test_r2 | 0.738435 |
| temperature_regression | pytorch-standard | pt_dropout_04 | test_r2 | 0.717495 |
| temperature_regression | scratch-layers | scratch_temp_residual_128x2 | val_r2 | 0.725996 |
| temperature_regression | scratch-layers | scratch_temp_flex_256_128_64 | val_r2 | 0.723149 |
| temperature_regression | scratch-layers | scratch_temp_flex_128_64 | val_r2 | 0.714586 |
| temperature_regression | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | r2 | 0.73211 |
| temperature_regression | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | r2 | 0.730551 |
| temperature_regression | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | r2 | 0.728414 |
| temperature_regression | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | r2 | 0.727887 |
| temperature_regression | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | r2 | 0.727168 |
| temperature_regression | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | r2 | 0.727082 |
| temperature_regression | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | r2 | 0.727007 |
| temperature_regression | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | r2 | 0.726801 |
| temperature_regression | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | r2 | 0.726774 |
| temperature_regression | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | r2 | 0.726603 |
| temperature_regression | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | r2 | 0.726483 |
| temperature_regression | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | r2 | 0.723102 |
| temperature_regression | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | r2 | 0.722418 |
| temperature_regression | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | r2 | 0.719058 |
| temperature_regression | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | r2 | 0.718457 |

### weather_classification

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| weather_classification | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | auc_score | 0.709709 |
| weather_classification | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | auc_score | 0.654068 |
| weather_classification | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | auc_score | 0.651434 |
| weather_classification | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | auc_score | 0.631949 |
| weather_classification | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | auc_score | 0.631625 |
| weather_classification | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | auc_score | 0.627354 |
| weather_classification | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | auc_score | 0.622279 |
| weather_classification | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | auc_score | 0.621468 |
| weather_classification | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | auc_score | 0.620157 |
| weather_classification | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | auc_score | 0.612195 |
| weather_classification | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | auc_score | 0.609576 |
| weather_classification | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | auc_score | 0.603718 |
| weather_classification | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | auc_score | 0.603668 |
| weather_classification | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | auc_score | 0.501536 |
| weather_classification | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | auc_score | 0.49985 |

### wind_forecasting

| task | family | config | metric | value |
| --- | --- | --- | --- | --- |
| wind_forecasting | pytorch-standard | pt_scheduler_cosine | val_r2 | 0.956552 |
| wind_forecasting | pytorch-standard | pt_baseline | val_r2 | 0.956439 |
| wind_forecasting | pytorch-standard | pt_scheduler_plateau | val_r2 | 0.956427 |
| wind_forecasting | pytorch-standard | pt_full_modern | val_r2 | 0.956344 |
| wind_forecasting | pytorch-standard | pt_adamw | val_r2 | 0.956331 |
| wind_forecasting | pytorch-standard | pt_sgd_momentum | val_r2 | 0.955246 |
| wind_forecasting | pytorch-standard | pt_dropout_02 | val_r2 | 0.955195 |
| wind_forecasting | pytorch-standard | pt_dropout_04 | val_r2 | 0.954682 |
| wind_forecasting | pytorch-standard | pt_bn_dropout | val_r2 | 0.950839 |
| wind_forecasting | pytorch-standard | pt_batchnorm | val_r2 | 0.947081 |
| wind_forecasting | scratch-layers | scratch_wind_flex_256_128 | val_r2 | 0.956077 |
| wind_forecasting | scratch-layers | scratch_wind_residual_128x2 | val_r2 | 0.949859 |
| wind_forecasting | standard-search | MLP(256x128,relu,a=0.0001,lr=0.01) | r2 | 0.971094 |
| wind_forecasting | standard-search | MLP(128x64,relu,a=0.0001,lr=0.01) | r2 | 0.971042 |
| wind_forecasting | standard-search | MLP(256x128x64,tanh,a=0.0001,lr=0.001) | r2 | 0.970946 |
| wind_forecasting | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.01) | r2 | 0.970887 |
| wind_forecasting | standard-search | MLP(128x64,relu,a=0.01,lr=0.001) | r2 | 0.970739 |
| wind_forecasting | standard-search | MLP(64,relu,a=0.0001,lr=0.001) | r2 | 0.970638 |
| wind_forecasting | standard-search | MLP(128x64,tanh,a=0.0001,lr=0.001) | r2 | 0.970584 |
| wind_forecasting | standard-search | MLP(256x128,relu,a=0.001,lr=0.001) | r2 | 0.970579 |
| wind_forecasting | standard-search | MLP(256x128,relu,a=0.0001,lr=0.001) | r2 | 0.970555 |
| wind_forecasting | standard-search | MLP(256x128x64,relu,a=0.0001,lr=0.001) | r2 | 0.970523 |
| wind_forecasting | standard-search | MLP(128x64,relu,a=0.001,lr=0.001) | r2 | 0.970411 |
| wind_forecasting | standard-search | MLP(128x64x32,relu,a=0.0001,lr=0.001) | r2 | 0.970371 |
| wind_forecasting | standard-search | MLP(128,relu,a=0.0001,lr=0.001) | r2 | 0.970303 |
| wind_forecasting | standard-search | MLP(128x64,relu,a=0.0001,lr=0.001) | r2 | 0.970011 |
| wind_forecasting | standard-search | MLP(256x128,tanh,a=0.0001,lr=0.001) | r2 | 0.969809 |

## Scratch-Layer Results

### Heart

| config | description | lr | weight_decay | batch_size | max_epochs | patience | val_auc | test_auc | test_acc | test_f1 | epochs_run | time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch_flex_small | ScratchFlexMLP hidden=[64,32], dropout=0.1 | 0.01 | 0.001 | 32 | 80 | 12 | 0.953685 | 0.942035 | 0.89916 | 0.905512 | 35 | 4.9 |
| scratch_flex_256_128 | ScratchFlexMLP hidden=[256,128], dropout=0.0 | 0.01 | 0.0001 | 32 | 120 | 16 | 0.953355 | 0.940051 | 0.865546 | 0.873016 | 30 | 3 |
| scratch_residual_128x2 | ScratchResidualMLP hidden=128, blocks=2, dropout=0.1 | 0.005 | 0.001 | 32 | 120 | 16 | 0.953465 | 0.928855 | 0.878151 | 0.88716 | 28 | 8 |

### Temperature

| config | description | lr | weight_decay | batch_size | max_epochs | patience | val_r2 | test_r2 | test_mae | test_rmse | epochs_run | time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch_temp_residual_128x2 | ScratchResidualMLP hidden=128, blocks=2, dropout=0.1 | 0.001 | 0.001 | 256 | 80 | 10 | 0.725996 | 0.7347 | 3.8629 | 4.9447 | 62 | 78.9 |
| scratch_temp_flex_256_128_64 | ScratchFlexMLP hidden=[256,128,64], dropout=0.0 | 0.001 | 0.0001 | 256 | 80 | 10 | 0.723149 | 0.733373 | 3.8752 | 4.957 | 67 | 34.8 |
| scratch_temp_flex_128_64 | ScratchFlexMLP hidden=[128,64], dropout=0.1, BN | 0.001 | 0.001 | 256 | 70 | 10 | 0.714586 | 0.725603 | 3.9773 | 5.0287 | 60 | 44.8 |

### Wind

| config | description | lr | weight_decay | batch_size | max_epochs | patience | val_r2 | test_r2 | test_mae | test_rmse | epochs_run | time_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| scratch_wind_flex_256_128 | ScratchFlexMLP hidden=[256,128], mirrors baseline width | 0.01 | 0.0001 | 512 | 90 | 12 | 0.956077 | 0.970979 | 135.515 | 228.6167 | 26 | 20.9 |
| scratch_wind_residual_128x2 | ScratchResidualMLP hidden=128, blocks=2, dropout=0.1 | 0.005 | 0.001 | 512 | 90 | 12 | 0.949859 | 0.964871 | 180.2006 | 251.5272 | 40 | 74.7 |

## Scratch Comparison Against Existing PyTorch Models

| task | metric | pt_adamw_config | pt_adamw_value | adv_residual_config | adv_residual_value | scratch_best_config | scratch_best_value | overall_best_label | overall_best_value |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_classification | test_auc | pt_adamw | 0.948484 | adv_residual_wide | 0.951672 | scratch_flex_small | 0.942035 | adv_residual_wide | 0.951672 |
| temperature_regression | test_r2 | pt_adamw | 0.748625 | adv_residual_wide | 0.752036 | scratch_temp_residual_128x2 | 0.7347 | adv_residual_wide | 0.752036 |
| wind_forecasting | test_r2 | pt_adamw | 0.97082 | n/a | n/a | scratch_wind_flex_256_128 | 0.970979 | pt_scheduler_cosine | 0.971016 |

## Files to Show the Professor

- `scratch_architectures/best_models_overview.csv`
- `scratch_architectures/architecture_catalog.csv`
- `pytorch_results/scratch_vs_pytorch_comparison.csv`
- `pytorch_results/scratch_layers_heart_results.csv`
- `pytorch_results/scratch_layers_temperature_results.csv`
- `pytorch_results/scratch_layers_wind_results.csv`
- `pytorch_results/heart_pytorch_results.csv`
- `pytorch_results/heart_advanced_results.csv`
- `pytorch_results/temperature_pytorch_results.csv`
- `pytorch_results/temperature_advanced_results.csv`
- `pytorch_results/wind_pytorch_results.csv`

## Run Commands

```powershell
& .venv\Scripts\Activate.ps1
& "c:/Users/LENOVO/Desktop/Sem 8 TPs/deepl/ML tp0/.venv/Scripts/python.exe" scratch_architectures/run_all.py --full
```