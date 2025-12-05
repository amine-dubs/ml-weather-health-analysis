import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')


# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Wind Turbine Scada dataset/T1.csv')


# Use all features except Date/Time as predictors, target is LV ActivePower (kW)
feature_columns = [col for col in df.columns if col != 'Date/Time']
target_column = 'LV ActivePower (kW)'

# Univariate: use only the target column for forecasting
data = df[target_column].values.reshape(-1, 1)


# Test both scalers
scalers = {
    'MinMaxScaler': MinMaxScaler(feature_range=(0, 1)),
    'StandardScaler': StandardScaler()
}


def create_sequences(data, window_size):
    """
    Create sequences for time series forecasting.
    
    Parameters:
    - data: The time series data
    - window_size: Number of past time steps to use as input
    
    Returns:
    - X: Input sequences
    - y: Target values
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def evaluate_model(y_true, y_pred, model_name, window_size, scaler, scaler_name):
    """Calculate and print evaluation metrics."""
    # Inverse transform to get actual values
    y_true_actual = scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    mse = mean_squared_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    r2 = r2_score(y_true_actual, y_pred_actual)
    mape = np.mean(np.abs((y_true_actual - y_pred_actual) / (y_true_actual + 1e-10))) * 100
    
    print(f"\n{model_name} - Window Size: {window_size} - Scaler: {scaler_name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    return {'model': model_name, 'window_size': window_size, 'scaler': scaler_name,
            'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


def plot_predictions(y_true, y_pred, model_name, window_size, scaler, scaler_name, sample_size=500):
    """Plot actual vs predicted values."""
    # Inverse transform to get actual values
    y_true_actual = scaler.inverse_transform(y_true[:sample_size].reshape(-1, 1))
    y_pred_actual = scaler.inverse_transform(y_pred[:sample_size].reshape(-1, 1))
    
    plt.figure(figsize=(15, 5))
    plt.plot(y_true_actual, label='Actual', alpha=0.7, linewidth=1)
    plt.plot(y_pred_actual, label='Predicted', alpha=0.7, linewidth=1)
    plt.title(f'{model_name} - Window: {window_size} - Scaler: {scaler_name} (First {sample_size} predictions)')
    plt.xlabel('Time Steps')
    plt.ylabel(f'{target_column}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'forecast_{model_name.replace(" ", "_")}_window_{window_size}_{scaler_name}.png', dpi=300, bbox_inches='tight')
    plt.close()


# Define different window sizes to test
window_sizes = [24]  # 1 hour, 2 hours, 4 hours, 8 hours, 16 hours (10-min intervals)

# Store results
all_results = []
# Store trained models
trained_models = {}

print("\n" + "="*80)
print("UNIVARIATE FORECASTING WITH DIFFERENT WINDOW SIZES AND SCALERS")
print("="*80)

# Train-test split ratio
train_ratio = 0.8

# Test different scalers, window sizes, and models
for scaler_name, scaler in scalers.items():
    print(f"\n{'#'*80}")
    print(f"TESTING SCALER: {scaler_name}")
    print(f"{'#'*80}")
    
    # Scale the data with current scaler
    data_scaled = scaler.fit_transform(data)
    train_size = int(len(data_scaled) * train_ratio)
    
    print(f"\nTotal samples: {len(data_scaled)}")
    print(f"Train samples: {train_size}")
    print(f"Test samples: {len(data_scaled) - train_size}")
    
    for window_size in window_sizes:
        print(f"\n{'='*80}")
        print(f"TESTING WINDOW SIZE: {window_size} (representing {window_size * 10} minutes)")
        print(f"{'='*80}")
        
        # Create sequences
        X, y = create_sequences(data_scaled, window_size)
        
        # Reshape X for sklearn models (flatten the window)
        X_flat = X.reshape(X.shape[0], -1)
        
        # Split into train and test
        split_idx = int(len(X_flat) * train_ratio)
        X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"\nSequences created:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  y_test shape: {y_test.shape}")
        
        # Model 1: Linear Regression
        print("\nTraining Linear Regression...")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)
        result_lr = evaluate_model(y_test, y_pred_lr, "Linear Regression", window_size, scaler, scaler_name)
        all_results.append(result_lr)
        plot_predictions(y_test, y_pred_lr, "Linear_Regression", window_size, scaler, scaler_name)
        # Store the trained model
        model_key = f"LinearRegression_w{window_size}_{scaler_name}"
        trained_models[model_key] = {
            'model': lr_model,
            'scaler': scaler,
            'window_size': window_size,
            'scaler_name': scaler_name,
            'metrics': result_lr
        }

        # Model 1b: Ridge Regression
        print("\nTraining Ridge Regression...")
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        y_pred_ridge = ridge_model.predict(X_test)
        result_ridge = evaluate_model(y_test, y_pred_ridge, "Ridge Regression", window_size, scaler, scaler_name)
        all_results.append(result_ridge)
        plot_predictions(y_test, y_pred_ridge, "Ridge_Regression", window_size, scaler, scaler_name)
        # Store the trained model
        model_key = f"RidgeRegression_w{window_size}_{scaler_name}"
        trained_models[model_key] = {
            'model': ridge_model,
            'scaler': scaler,
            'window_size': window_size,
            'scaler_name': scaler_name,
            'metrics': result_ridge
        }
        
        # Model 2: Random Forest
        print("\nTraining Random Forest...")
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        rf_model.fit(X_train, y_train.ravel())
        y_pred_rf = rf_model.predict(X_test)
        result_rf = evaluate_model(y_test, y_pred_rf, "Random Forest", window_size, scaler, scaler_name)
        all_results.append(result_rf)
        plot_predictions(y_test, y_pred_rf, "Random_Forest", window_size, scaler, scaler_name)
        
        # Store the trained model
        model_key = f"RandomForest_w{window_size}_{scaler_name}"
        trained_models[model_key] = {
            'model': rf_model,
            'scaler': scaler,
            'window_size': window_size,
            'scaler_name': scaler_name,
            'metrics': result_rf
        }
        
        # Model 3: XGBoost
        print("\nTraining XGBoost...")
        xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                 random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train.ravel())
        y_pred_xgb = xgb_model.predict(X_test)
        result_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost", window_size, scaler, scaler_name)
        all_results.append(result_xgb)
        plot_predictions(y_test, y_pred_xgb, "XGBoost", window_size, scaler, scaler_name)
        
        # Store the trained model
        model_key = f"XGBoost_w{window_size}_{scaler_name}"
        trained_models[model_key] = {
            'model': xgb_model,
            'scaler': scaler,
            'window_size': window_size,
            'scaler_name': scaler_name,
            'metrics': result_xgb
        }

# Create summary results
print("\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)

results_df = pd.DataFrame(all_results)
print("\n", results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('forecasting_results_summary.csv', index=False)
print("\nResults saved to 'forecasting_results_summary.csv'")

# Find best model for each metric
print("\n" + "="*80)
print("BEST MODELS BY METRIC")
print("="*80)

best_rmse = results_df.loc[results_df['rmse'].idxmin()]
best_mae = results_df.loc[results_df['mae'].idxmin()]
best_r2 = results_df.loc[results_df['r2'].idxmax()]
best_mape = results_df.loc[results_df['mape'].idxmin()]

print(f"\nBest RMSE: {best_rmse['model']} (Window: {best_rmse['window_size']}, Scaler: {best_rmse['scaler']}) - RMSE: {best_rmse['rmse']:.4f}")
print(f"Best MAE: {best_mae['model']} (Window: {best_mae['window_size']}, Scaler: {best_mae['scaler']}) - MAE: {best_mae['mae']:.4f}")
print(f"Best R2: {best_r2['model']} (Window: {best_r2['window_size']}, Scaler: {best_r2['scaler']}) - R2: {best_r2['r2']:.4f}")
print(f"Best MAPE: {best_mape['model']} (Window: {best_mape['window_size']}, Scaler: {best_mape['scaler']}) - MAPE: {best_mape['mape']:.4f}%")

# Save the best model (based on R2 score)
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

best_model_name = best_r2['model'].replace(" ", "")
best_window = best_r2['window_size']
best_scaler_name = best_r2['scaler']
model_key = f"{best_model_name}_w{best_window}_{best_scaler_name}"

best_model_info = trained_models[model_key]

# Save model, scaler, and configuration
model_save_dict = {
    'model': best_model_info['model'],
    'scaler': best_model_info['scaler'],
    'window_size': best_model_info['window_size'],
    'scaler_name': best_model_info['scaler_name'],
    'target_column': target_column,
    'metrics': best_model_info['metrics']
}

with open('best_model.pkl', 'wb') as f:
    pickle.dump(model_save_dict, f)

print(f"\nBest model saved to 'best_model.pkl'")
print(f"  Model: {best_r2['model']}")
print(f"  Window Size: {best_window}")
print(f"  Scaler: {best_scaler_name}")
print(f"  R2 Score: {best_r2['r2']:.4f}")
print(f"  RMSE: {best_r2['rmse']:.4f}")

# Create a text file with model information
with open('best_model_info.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BEST MODEL INFORMATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model Type: {best_r2['model']}\n")
    f.write(f"Window Size: {best_window} (representing {best_window * 10} minutes)\n")
    f.write(f"Scaler: {best_scaler_name}\n")
    f.write(f"Target Variable: {target_column}\n\n")
    f.write("Performance Metrics:\n")
    f.write(f"  - RMSE: {best_r2['rmse']:.4f}\n")
    f.write(f"  - MAE: {best_r2['mae']:.4f}\n")
    f.write(f"  - R² Score: {best_r2['r2']:.4f}\n")
    f.write(f"  - MAPE: {best_r2['mape']:.4f}%\n\n")
    f.write("="*80 + "\n")
    f.write("HOW TO USE THE SAVED MODEL\n")
    f.write("="*80 + "\n\n")
    f.write("import pickle\n")
    f.write("import numpy as np\n\n")
    f.write("# Load the model\n")
    f.write("with open('best_model.pkl', 'rb') as f:\n")
    f.write("    model_dict = pickle.load(f)\n\n")
    f.write("model = model_dict['model']\n")
    f.write("scaler = model_dict['scaler']\n")
    f.write(f"window_size = model_dict['window_size']  # {best_window}\n\n")
    f.write("# Prepare your input data (last 'window_size' time steps)\n")
    f.write("# input_data should be a numpy array of shape (window_size, 1)\n")
    f.write("input_data = your_data[-window_size:].reshape(-1, 1)\n\n")
    f.write("# Scale the input\n")
    f.write("input_scaled = scaler.transform(input_data)\n\n")
    f.write("# Flatten for prediction\n")
    f.write("input_flat = input_scaled.flatten().reshape(1, -1)\n\n")
    f.write("# Make prediction\n")
    f.write("prediction_scaled = model.predict(input_flat)\n\n")
    f.write("# Inverse transform to get actual value\n")
    f.write("prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))\n")
    f.write("print(f'Predicted {target_column}: {prediction[0][0]:.2f}')\n")

print("Model usage instructions saved to 'best_model_info.txt'")

# Plot comparison of window sizes for each model and scaler
plt.figure(figsize=(20, 12))

models = results_df['model'].unique()
metrics = ['rmse', 'mae', 'r2', 'mape']
metric_names = ['RMSE', 'MAE', 'R² Score', 'MAPE (%)']

for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names), 1):
    plt.subplot(2, 2, idx)
    for model in models:
        for scaler_name in results_df['scaler'].unique():
            model_scaler_data = results_df[(results_df['model'] == model) & (results_df['scaler'] == scaler_name)]
            plt.plot(model_scaler_data['window_size'], model_scaler_data[metric], 
                    marker='o', label=f'{model} ({scaler_name})', linewidth=2)
    
    plt.xlabel('Window Size')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Window Size')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show only our window sizes
    plt.xticks(window_sizes)

plt.tight_layout()
plt.savefig('window_size_comparison.png', dpi=300, bbox_inches='tight')
print("\nComparison plot saved to 'window_size_comparison.png'")

# Create a heatmap of results for each scaler
for scaler_name in results_df['scaler'].unique():
    scaler_results = results_df[results_df['scaler'] == scaler_name]
    pivot_rmse = scaler_results.pivot(index='model', columns='window_size', values='rmse')
    pivot_r2 = scaler_results.pivot(index='model', columns='window_size', values='r2')
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # RMSE heatmap
    im1 = axes[0].imshow(pivot_rmse.values, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xticks(range(len(pivot_rmse.columns)))
    axes[0].set_yticks(range(len(pivot_rmse.index)))
    axes[0].set_xticklabels(pivot_rmse.columns)
    axes[0].set_yticklabels(pivot_rmse.index)
    axes[0].set_xlabel('Window Size')
    axes[0].set_ylabel('Model')
    axes[0].set_title(f'RMSE Heatmap - {scaler_name} (Lower is Better)')
    plt.colorbar(im1, ax=axes[0])
    
    # Add text annotations
    for i in range(len(pivot_rmse.index)):
        for j in range(len(pivot_rmse.columns)):
            text = axes[0].text(j, i, f'{pivot_rmse.values[i, j]:.2f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    # R2 heatmap
    im2 = axes[1].imshow(pivot_r2.values, cmap='RdYlGn', aspect='auto')
    axes[1].set_xticks(range(len(pivot_r2.columns)))
    axes[1].set_yticks(range(len(pivot_r2.index)))
    axes[1].set_xticklabels(pivot_r2.columns)
    axes[1].set_yticklabels(pivot_r2.index)
    axes[1].set_xlabel('Window Size')
    axes[1].set_ylabel('Model')
    axes[1].set_title(f'R² Score Heatmap - {scaler_name} (Higher is Better)')
    plt.colorbar(im2, ax=axes[1])
    
    # Add text annotations
    for i in range(len(pivot_r2.index)):
        for j in range(len(pivot_r2.columns)):
            text = axes[1].text(j, i, f'{pivot_r2.values[i, j]:.4f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'performance_heatmap_{scaler_name}.png', dpi=300, bbox_inches='tight')
    print(f"Performance heatmap saved to 'performance_heatmap_{scaler_name}.png'")

print("\n" + "="*80)
print("FORECASTING COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. forecasting_results_summary.csv - Detailed results for all models, window sizes, and scalers")
print("  2. best_model.pkl - Saved best model with scaler and configuration")
print("  3. best_model_info.txt - Instructions on how to use the saved model")
print("  4. window_size_comparison.png - Line plots comparing metrics across window sizes and scalers")
print("  5. performance_heatmap_MinMaxScaler.png - Heatmaps for MinMaxScaler")
print("  6. performance_heatmap_StandardScaler.png - Heatmaps for StandardScaler")
print(f"  7. forecast_*.png - {len(all_results)} individual prediction plots")

# Print scaler comparison summary
print("\n" + "="*80)
print("SCALER COMPARISON SUMMARY")
print("="*80)
for scaler_name in results_df['scaler'].unique():
    scaler_results = results_df[results_df['scaler'] == scaler_name]
    print(f"\n{scaler_name}:")
    print(f"  Average RMSE: {scaler_results['rmse'].mean():.4f}")
    print(f"  Average MAE: {scaler_results['mae'].mean():.4f}")
    print(f"  Average R2: {scaler_results['r2'].mean():.4f}")
    print(f"  Average MAPE: {scaler_results['mape'].mean():.4f}%")
