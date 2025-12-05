import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('Wind Turbine Scada dataset/T1.csv')

# Configuration
target_column = 'LV ActivePower (kW)'
feature_columns = [col for col in df.columns if col != 'Date/Time']
horizon = 6       # Predict next 1 hour (6 * 10min)
window_size = 24  # Use past 4 hours data

print(f"\nTarget variable: {target_column}")
print(f"Features: {feature_columns}")
print(f"Predicting {horizon} steps ahead (next hour)")

# Separate features and target for scaling
# We need a scaler for Y specifically to inverse transform predictions later
X_data = df[feature_columns].values
y_data = df[[target_column]].values

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X_scaled = scaler_X.fit_transform(X_data)
y_scaled = scaler_y.fit_transform(y_data)

# Sequence creation for Multivariate Multi-step forecasting
def create_multivariate_multistep_sequences(X_data, y_data, window_size, horizon):
    """
    Input: History of ALL features (X)
    Output: Future of TARGET variable only (y)
    """
    X, y = [], []
    # Ensure we have enough data for window + horizon
    for i in range(len(X_data) - window_size - horizon + 1):
        # Input: window_size steps of all features
        X.append(X_data[i:i + window_size])
        # Output: horizon steps of target variable
        y.append(y_data[i + window_size:i + window_size + horizon].flatten())
    return np.array(X), np.array(y)

print("Creating sequences...")
X, y = create_multivariate_multistep_sequences(X_scaled, y_scaled, window_size, horizon)

# Flatten X for models: (n_samples, window_size * n_features)
n_samples, w, n_feats = X.shape
X_flat = X.reshape(n_samples, -1)

print(f"Input shape (original): {X.shape}")
print(f"Input shape (flattened): {X_flat.shape}")
print(f"Output shape: {y.shape}")

# Train-test split (chronological)
train_ratio = 0.8
split_idx = int(len(X_flat) * train_ratio)
X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")

# Models
# Note: XGBRegressor needs MultiOutputRegressor to predict multiple steps at once
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))
}

results = []

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Inverse transform predictions and actuals
    # y_test and y_pred are (n_samples, horizon)
    # We need to inverse transform them using the target scaler
    y_test_actual = scaler_y.inverse_transform(y_test)
    y_pred_actual = scaler_y.inverse_transform(y_pred)

    # Evaluate for each step ahead
    print(f"--- {model_name} Performance ---")
    for step in range(horizon):
        mse = mean_squared_error(y_test_actual[:, step], y_pred_actual[:, step])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual[:, step], y_pred_actual[:, step])
        r2 = r2_score(y_test_actual[:, step], y_pred_actual[:, step])
        mape = np.mean(np.abs((y_test_actual[:, step] - y_pred_actual[:, step]) / (y_test_actual[:, step] + 1e-10))) * 100
        
        print(f"Step {step+1} (+{(step+1)*10}m): RMSE={rmse:.2f}, R2={r2:.3f}")
        results.append({
            'model': model_name, 
            'step': step+1, 
            'rmse': rmse, 
            'mae': mae, 
            'r2': r2, 
            'mape': mape
        })

    # Plot predictions for a sample
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, horizon+1), y_test_actual[0], label='Actual', marker='o')
    plt.plot(range(1, horizon+1), y_pred_actual[0], label='Predicted', marker='x')
    plt.title(f'{model_name} - Next Hour Forecast (Sample 0)')
    plt.xlabel('Steps Ahead (10 min intervals)')
    plt.ylabel(target_column)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'multivariate_multistep_{model_name.replace(" ", "_")}.png', dpi=300)
    plt.close()

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('multivariate_multistep_results.csv', index=False)
print("\nResults saved to 'multivariate_multistep_results.csv'")

# Save best model (based on average R2 across all steps)
avg_r2 = results_df.groupby('model')['r2'].mean()
best_model_name = avg_r2.idxmax()
best_model = models[best_model_name]

print(f"\nBest Model: {best_model_name} (Avg R2: {avg_r2.max():.4f})")

with open('best_multivariate_multistep_model.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model, 
        'scaler_X': scaler_X, 
        'scaler_y': scaler_y, 
        'window_size': window_size, 
        'horizon': horizon,
        'feature_columns': feature_columns
    }, f)
print("Best model saved to 'best_multivariate_multistep_model.pkl'")
