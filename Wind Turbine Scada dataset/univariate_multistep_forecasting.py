import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Wind Turbine Scada dataset/T1.csv')

# Select the target variable (LV ActivePower)
target_column = 'LV ActivePower (kW)'
data = df[[target_column]].values

# Use only MinMaxScaler (based on previous results)
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Predict the next hour (6 steps ahead, since data is at 10-min intervals)
horizon = 6
window_size = 24  # Use 4 hours of history (can be tuned)

print(f"\nTarget variable: {target_column}")
print(f"Predicting {horizon} steps ahead (next hour)")

# Sequence creation for multi-step forecasting (direct strategy)
def create_multistep_sequences(data, window_size, horizon):
    X, y = [], []
    for i in range(len(data) - window_size - horizon + 1):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size:i + window_size + horizon].flatten())
    return np.array(X), np.array(y)

X, y = create_multistep_sequences(data_scaled, window_size, horizon)
X_flat = X.reshape(X.shape[0], -1)

# Train-test split (chronological)
train_ratio = 0.8
split_idx = int(len(X_flat) * train_ratio)
X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\nTotal samples: {len(X_flat)}")
print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Use only the best two models from previous experiment
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
}

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inverse transform for each step
    y_test_actual = scaler.inverse_transform(y_test)
    y_pred_actual = scaler.inverse_transform(y_pred)

    # Evaluate for each step ahead
    for step in range(horizon):
        mse = mean_squared_error(y_test_actual[:, step], y_pred_actual[:, step])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual[:, step], y_pred_actual[:, step])
        r2 = r2_score(y_test_actual[:, step], y_pred_actual[:, step])
        mape = np.mean(np.abs((y_test_actual[:, step] - y_pred_actual[:, step]) / (y_test_actual[:, step] + 1e-10))) * 100
        print(f"{model_name} - Step {step+1} (t+{(step+1)*10} min): RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.3f}, MAPE={mape:.2f}%")
        results.append({'model': model_name, 'step': step+1, 'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape})

    # Plot predictions for the first test sample
    plt.figure(figsize=(10, 4))
    plt.plot(range(horizon), y_test_actual[0], label='Actual', marker='o')
    plt.plot(range(horizon), y_pred_actual[0], label='Predicted', marker='o')
    plt.title(f'{model_name} - Next Hour Forecast (First Test Sample)')
    plt.xlabel('Step (10-min intervals)')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'forecast_{model_name.replace(" ", "_")}_multistep.png', dpi=300, bbox_inches='tight')
    plt.close()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('multistep_forecasting_results_summary.csv', index=False)
print("\nResults saved to 'multistep_forecasting_results_summary.csv'")

# Save the best model for the first step ahead
best_result = max(results, key=lambda x: x['r2'] if x['step'] == 1 else -np.inf)
best_model = models[best_result['model']]
with open('best_multistep_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'window_size': window_size, 'horizon': horizon, 'target_column': target_column}, f)
print(f"\nBest model for t+10min saved to 'best_multistep_model.pkl' ({best_result['model']})")
