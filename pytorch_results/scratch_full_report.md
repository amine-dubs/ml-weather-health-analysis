# Scratch Architectures — Aggregated Results

This report aggregates the primitive-layer 'from-scratch' experiments and comparisons against existing PyTorch configs.

## Main comparison (summary)

| task                   | metric   | pt_adamw_config   |   pt_adamw_value | adv_residual_config   |   adv_residual_value | scratch_best_config         |   scratch_best_value | overall_best_label   |   overall_best_value |
|:-----------------------|:---------|:------------------|-----------------:|:----------------------|---------------------:|:----------------------------|---------------------:|:---------------------|---------------------:|
| heart_classification   | test_auc | pt_adamw          |         0.948484 | adv_residual_wide     |             0.951672 | scratch_flex_small          |             0.942035 | adv_residual_wide    |             0.951672 |
| temperature_regression | test_r2  | pt_adamw          |         0.748625 | adv_residual_wide     |             0.752036 | scratch_temp_residual_128x2 |             0.7347   | adv_residual_wide    |             0.752036 |
| wind_forecasting       | test_r2  | pt_adamw          |         0.97082  | nan                   |           nan        | scratch_wind_flex_256_128   |             0.970979 | pt_scheduler_cosine  |             0.971016 |


![Main comparison](pytorch_results\plots\scratch_vs_pytorch_main_metric.png)

![Details](pytorch_results\plots\scratch_other_tasks_detail.png)

## Heart — scratch results

| config                 | description                                          |    lr |   weight_decay |   batch_size |   max_epochs |   patience |   val_auc |   test_auc |   test_acc |   test_f1 |   epochs_run |   time_s |
|:-----------------------|:-----------------------------------------------------|------:|---------------:|-------------:|-------------:|-----------:|----------:|-----------:|-----------:|----------:|-------------:|---------:|
| scratch_flex_small     | ScratchFlexMLP hidden=[64,32], dropout=0.1           | 0.01  |         0.001  |           32 |           80 |         12 |  0.953685 |   0.942035 |   0.89916  |  0.905512 |           35 |      4.9 |
| scratch_flex_256_128   | ScratchFlexMLP hidden=[256,128], dropout=0.0         | 0.01  |         0.0001 |           32 |          120 |         16 |  0.953355 |   0.940051 |   0.865546 |  0.873016 |           30 |      3   |
| scratch_residual_128x2 | ScratchResidualMLP hidden=128, blocks=2, dropout=0.1 | 0.005 |         0.001  |           32 |          120 |         16 |  0.953465 |   0.928855 |   0.878151 |  0.88716  |           28 |      8   |


## Temp — scratch results

| config                       | description                                          |    lr |   weight_decay |   batch_size |   max_epochs |   patience |   val_r2 |   test_r2 |   test_mae |   test_rmse |   epochs_run |   time_s |
|:-----------------------------|:-----------------------------------------------------|------:|---------------:|-------------:|-------------:|-----------:|---------:|----------:|-----------:|------------:|-------------:|---------:|
| scratch_temp_residual_128x2  | ScratchResidualMLP hidden=128, blocks=2, dropout=0.1 | 0.001 |         0.001  |          256 |           80 |         10 | 0.725996 |  0.7347   |     3.8629 |      4.9447 |           62 |     78.9 |
| scratch_temp_flex_256_128_64 | ScratchFlexMLP hidden=[256,128,64], dropout=0.0      | 0.001 |         0.0001 |          256 |           80 |         10 | 0.723149 |  0.733373 |     3.8752 |      4.957  |           67 |     34.8 |
| scratch_temp_flex_128_64     | ScratchFlexMLP hidden=[128,64], dropout=0.1, BN      | 0.001 |         0.001  |          256 |           70 |         10 | 0.714586 |  0.725603 |     3.9773 |      5.0287 |           60 |     44.8 |


## Wind — scratch results

| config                      | description                                             |    lr |   weight_decay |   batch_size |   max_epochs |   patience |   val_r2 |   test_r2 |   test_mae |   test_rmse |   epochs_run |   time_s |
|:----------------------------|:--------------------------------------------------------|------:|---------------:|-------------:|-------------:|-----------:|---------:|----------:|-----------:|------------:|-------------:|---------:|
| scratch_wind_flex_256_128   | ScratchFlexMLP hidden=[256,128], mirrors baseline width | 0.01  |         0.0001 |          512 |           90 |         12 | 0.956077 |  0.970979 |    135.515 |     228.617 |           26 |     20.9 |
| scratch_wind_residual_128x2 | ScratchResidualMLP hidden=128, blocks=2, dropout=0.1    | 0.005 |         0.001  |          512 |           90 |         12 | 0.949859 |  0.964871 |    180.201 |     251.527 |           40 |     74.7 |

