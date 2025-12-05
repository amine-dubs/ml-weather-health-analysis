import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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

# Multivariate: use ALL columns (including target) as features for the window
# This creates proper multivariate time series: [power_history + weather_history] -> next_power
all_features = feature_columns  # Includes target + weather features

# MinMax normalization - normalize all features together
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[all_features])

print(f"\nTarget variable: {target_column}")
print(f"All features (including target): {all_features}")
print(f"Number of features: {len(all_features)}")

# Use a single window size (best from previous: 24)
window_size = 24  # 4 hours (matching univariate)

print(f"Window size: {window_size}")

# Sequence creation for TRUE multivariate forecasting
# Input: window of [power + weather features] -> Output: next power value
def create_multivariate_sequences(data, window_size, target_idx=0):
    """
    Create sequences where input is a window of ALL features (power + weather)
    and output is the next power value.
    
    Args:
        data: Scaled data with all features (power is column 0)
        window_size: Number of time steps to look back
        target_idx: Index of target column (default 0 for power)
    """
    X_seq, y_seq = [], []
    for i in range(len(data) - window_size):
        # Input: window of ALL features (power + weather)
        X_seq.append(data[i:i + window_size].flatten())
        # Output: next power value only
        y_seq.append(data[i + window_size, target_idx])
    return np.array(X_seq), np.array(y_seq)

# Target is first column (LV ActivePower)
target_idx = all_features.index(target_column)
X_seq, y_seq = create_multivariate_sequences(data_scaled, window_size, target_idx)

# Train-test split (chronological)
train_ratio = 0.8
split_idx = int(len(X_seq) * train_ratio)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"\nTotal samples: {len(X_seq)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Use only the best two models from previous experiment
from sklearn.linear_model import LinearRegression, Ridge
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train.ravel())
    y_pred = model.predict(X_test)

    # Inverse transform - need to reconstruct full feature array for inverse scaling
    # Create dummy arrays with just the target column at correct position
    y_test_full = np.zeros((len(y_test), len(all_features)))
    y_test_full[:, target_idx] = y_test.ravel()
    y_test_actual = scaler.inverse_transform(y_test_full)[:, target_idx]
    
    y_pred_full = np.zeros((len(y_pred), len(all_features)))
    y_pred_full[:, target_idx] = y_pred.ravel()
    y_pred_actual = scaler.inverse_transform(y_pred_full)[:, target_idx]

    mse = mean_squared_error(y_test_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)
    mape = np.mean(np.abs((y_test_actual - y_pred_actual) / (y_test_actual + 1e-10))) * 100

    print(f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}, MAPE={mape:.2f}%")
    results.append({'model': model_name, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape})

    # Plot predictions for the first 500 test samples
    plt.figure(figsize=(15, 5))
    plt.plot(y_test_actual[:500], label='Actual', alpha=0.7, linewidth=1)
    plt.plot(y_pred_actual[:500], label='Predicted', alpha=0.7, linewidth=1)
    plt.title(f'{model_name} - Next Step Forecast (First 500 test samples)')
    plt.xlabel('Time Steps')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'multivariate_forecast_{model_name.replace(" ", "_")}_nextstep.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('multivariate_forecasting_results_summary.csv', index=False)
print("\nResults saved to 'multivariate_forecasting_results_summary.csv'")

# Save the best model
best_result = max(results, key=lambda x: x['r2'])
best_model = models[best_result['model']]
with open('best_multivariate_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model, 
        'scaler': scaler,  # Single scaler for all features
        'window_size': window_size, 
        'all_features': all_features,  # All features including target
        'target_column': target_column,
        'target_idx': target_idx
    }, f)
print(f"\nBest multivariate model saved to 'best_multivariate_model.pkl' ({best_result['model']})")
print(f"Model uses {len(all_features)} features (including power history) × {window_size} time steps = {len(all_features) * window_size} input features")
