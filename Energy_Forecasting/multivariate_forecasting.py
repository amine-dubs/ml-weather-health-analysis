import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')


# Load the dataset robustly using absolute path
print("Loading dataset...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '..', 'Forecasting', 'Forecasting', 'EnergyConsuption', 'pjm_hourly_est.csv')
df = pd.read_csv(os.path.normpath(data_path))

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names: {df.columns.tolist()}")

# Convert Datetime column to datetime
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Sort by datetime
df = df.sort_values('Datetime').reset_index(drop=True)

# Filter to use only data where PJME is filled (from 2002 onwards - much more data than PJM_Load)
df = df[df['PJME'].notnull()].copy()

print(f"\nFiltered data shape: {df.shape}")
print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
print(f"Note: Using PJME which has data through 2018, not PJM_Load which stops in 2002")

# Drop Datetime column for modeling
df_model = df.drop('Datetime', axis=1)

# Check for any remaining missing values
print(f"\nMissing values per column:")
print(df_model.isnull().sum())

# Drop columns with all missing values
df_model = df_model.dropna(axis=1, how='all')

# Fill any remaining missing values with forward fill
df_model = df_model.fillna(method='ffill').fillna(method='bfill')

print(f"\nFinal data shape: {df_model.shape}")
print(f"Columns: {df_model.columns.tolist()}")

# Set target column
target_column = 'PJME'

# Get feature columns (all except target)
feature_columns = [col for col in df_model.columns if col != target_column]

print(f"\nTarget: {target_column}")
print(f"Features: {feature_columns}")

# Prepare data
data = df_model.values

def create_multivariate_sequences(data, window_size, target_idx):
    """
    Create multivariate sequences for time series forecasting.
    
    Parameters:
    - data: The multivariate time series data
    - window_size: Number of past time steps to use as input
    - target_idx: Index of the target column
    
    Returns:
    - X: Input sequences (all features)
    - y: Target values (only target column)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


def evaluate_model(y_true, y_pred, model_name, window_size, target_scaler):
    """Calculate and print evaluation metrics."""
    # Inverse transform to get actual values
    y_true_actual = target_scaler.inverse_transform(y_true.reshape(-1, 1))
    y_pred_actual = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    
    mse = mean_squared_error(y_true_actual, y_pred_actual)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_actual, y_pred_actual)
    r2 = r2_score(y_true_actual, y_pred_actual)
    mape = np.mean(np.abs((y_true_actual - y_pred_actual) / (y_true_actual + 1e-10))) * 100
    
    print(f"\n{model_name} - Window Size: {window_size}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R2 Score: {r2:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    
    return {'model': model_name, 'window_size': window_size,
            'rmse': rmse, 'mae': mae, 'r2': r2, 'mape': mape}


# Window sizes to test
window_sizes = [24, 168]  # 1 day, 1 week

# Store results
all_results = []

print("\n" + "="*80)
print("MULTIVARIATE FORECASTING")
print("="*80)

# Train-test split ratio
train_ratio = 0.8

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Create separate scaler for target column
target_idx = df_model.columns.tolist().index(target_column)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler.fit(data[:, target_idx].reshape(-1, 1))

train_size = int(len(data_scaled) * train_ratio)

print(f"\nTotal samples: {len(data_scaled)}")
print(f"Train samples: {train_size}")
print(f"Test samples: {len(data_scaled) - train_size}")

for window_size in window_sizes:
    print(f"\n{'='*80}")
    print(f"TESTING WINDOW SIZE: {window_size} hours")
    print(f"{'='*80}")
    
    # Create sequences
    X, y = create_multivariate_sequences(data_scaled, window_size, target_idx)
    
    # Reshape X for sklearn models (flatten all features and time steps)
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
    
    # Model 1: Ridge Regression
    print("\nTraining Ridge Regression...")
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    result_ridge = evaluate_model(y_test, y_pred_ridge, "Ridge Regression", window_size, target_scaler)
    all_results.append(result_ridge)
    
    # Model 2: XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, 
                             random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    result_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost", window_size, target_scaler)
    all_results.append(result_xgb)

# Create summary results
print("\n" + "="*80)
print("SUMMARY OF ALL RESULTS")
print("="*80)

results_df = pd.DataFrame(all_results)
print("\n", results_df.to_string(index=False))

# Save results to CSV
results_df.to_csv('multivariate_forecasting_results_summary.csv', index=False)
print("\nResults saved to 'multivariate_forecasting_results_summary.csv'")


# Find best model
best_r2 = results_df.loc[results_df['r2'].idxmax()]

print("\n" + "="*80)
print("BEST MODEL")
print("="*80)
print(f"\nModel: {best_r2['model']}")
print(f"Window Size: {best_r2['window_size']}")
print(f"R2 Score: {best_r2['r2']:.4f}")
print(f"RMSE: {best_r2['rmse']:.4f}")
print(f"MAE: {best_r2['mae']:.4f}")
print(f"MAPE: {best_r2['mape']:.4f}%")


# Save the best model (univariate style)
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

best_model_name = best_r2['model'].replace(" ", "")
best_window = best_r2['window_size']
model_key = f"{best_model_name}_w{best_window}"

# Re-train the best model on the full training data for the best window size
X, y = create_multivariate_sequences(data_scaled, int(best_window), target_idx)
X_flat = X.reshape(X.shape[0], -1)
y_flat = y
split_idx = int(len(X_flat) * train_ratio)
X_train, X_test = X_flat[:split_idx], X_flat[split_idx:]
y_train, y_test = y_flat[:split_idx], y_flat[split_idx:]

if best_r2['model'] == "Ridge Regression":
    best_model = Ridge(alpha=1.0)
elif best_r2['model'] == "Random Forest":
    best_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
elif best_r2['model'] == "XGBoost":
    best_model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
else:
    raise ValueError(f"Unknown model: {best_r2['model']}")

best_model.fit(X_train, y_train)

# Save model, scalers, and configuration
model_save_dict = {
    'model': best_model,
    'scaler': scaler,
    'target_scaler': target_scaler,
    'window_size': best_window,
    'target_column': target_column,
    'feature_columns': feature_columns,
    'metrics': best_r2.to_dict()
}

with open('pjm_energy_multivariate_best_model.pkl', 'wb') as f:
    pickle.dump(model_save_dict, f)

print(f"\nBest model saved to 'pjm_energy_multivariate_best_model.pkl'")
print(f"  Model: {best_r2['model']}")
print(f"  Window Size: {best_window}")
print(f"  R2 Score: {best_r2['r2']:.4f}")
print(f"  RMSE: {best_r2['rmse']:.4f}")

# Create a text file with model information
with open('pjm_energy_multivariate_best_model_info.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BEST MULTIVARIATE MODEL INFORMATION\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model Type: {best_r2['model']}\n")
    f.write(f"Window Size: {best_window} hours\n")
    f.write(f"Target Variable: {target_column}\n")
    f.write(f"Features: {', '.join(feature_columns)}\n\n")
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
    f.write("with open('pjm_energy_multivariate_best_model.pkl', 'rb') as f:\n")
    f.write("    model_dict = pickle.load(f)\n\n")
    f.write("model = model_dict['model']\n")
    f.write("scaler = model_dict['scaler']\n")
    f.write("target_scaler = model_dict['target_scaler']\n")
    f.write(f"window_size = model_dict['window_size']  # {best_window}\n")
    f.write(f"feature_columns = model_dict['feature_columns']\n\n")
    f.write("# Prepare your input data (last 'window_size' time steps, all features)\n")
    f.write("# input_data should be a numpy array of shape (window_size, n_features)\n")
    f.write("input_data = your_data[-window_size:]\n\n")
    f.write("# Scale the input\n")
    f.write("input_scaled = scaler.transform(input_data)\n\n")
    f.write("# Flatten for prediction\n")
    f.write("input_flat = input_scaled.flatten().reshape(1, -1)\n\n")
    f.write("# Make prediction\n")
    f.write("prediction_scaled = model.predict(input_flat)\n\n")
    f.write("# Inverse transform to get actual value\n")
    f.write("prediction = target_scaler.inverse_transform(np.array(prediction_scaled).reshape(-1, 1))\n")
    f.write(f"print(f'Predicted {target_column}: {{prediction[0][0]:.2f}}')\n")

print("Model usage instructions saved to 'pjm_energy_multivariate_best_model_info.txt'")

print("\n" + "="*80)
print("MULTIVARIATE FORECASTING COMPLETE!")
print("="*80)
