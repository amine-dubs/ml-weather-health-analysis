import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
from joblib import dump
import json
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("MULTI-OUTPUT REGRESSION: PREDICTING PRESSURE & HUMIDITY")
print("="*70)

# Create directories for results
if not os.path.exists('multi_output_results'):
    os.makedirs('multi_output_results')
if not os.path.exists('multi_output_results/models'):
    os.makedirs('multi_output_results/models')

# Check if model already exists
model_exists = os.path.exists('multi_output_results/models/best_model.joblib')
if model_exists:
    print("\n" + "!"*70)
    print("MODEL ALREADY EXISTS!")
    print("!"*70)
    print("\nFound existing model at: multi_output_results/models/best_model.joblib")
    print("\nOptions:")
    print("  1. Delete the model file to retrain")
    print("  2. Use the existing model")
    print("\nExiting to avoid retraining...")
    print("="*70)
    exit(0)

# --- 1. LOAD AND PREPROCESS DATA ---
print("\n1. LOADING AND PREPROCESSING DATA")
print("=" * 70)

# Load the dataset
df = pd.read_csv('Dataset1.csv')
print(f"Original dataset shape: {df.shape}")

# Drop only non-feature columns (keep Apparent Temperature as a feature)
columns_to_drop = ['Formatted Date', 'Daily Summary']
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")
print(f"Dataset shape after dropping columns: {df.shape}")

# Define target variables (Pressure and Humidity)
target_columns = ['Pressure (millibars)', 'Humidity']

# Separate features and targets
X = df.drop(target_columns, axis=1)
y = df[target_columns]

print(f"\nTarget variables: {target_columns}")
print(f"Features shape: {X.shape}")
print(f"Targets shape: {y.shape}")

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"\nCategorical features: {list(categorical_features)}")
print(f"Numerical features: {list(numerical_features)}")

# Check for missing values in targets
print(f"\nMissing values in targets:")
print(f"  Pressure (millibars): {y['Pressure (millibars)'].isnull().sum()}")
print(f"  Humidity: {y['Humidity'].isnull().sum()}")

# Handle zero pressure values (sensor errors)
zero_pressure_count = (y['Pressure (millibars)'] == 0).sum()
if zero_pressure_count > 0:
    print(f"\nFound {zero_pressure_count} zero pressure values (sensor errors)")
    print("Converting zero pressure to NaN for imputation...")
    y.loc[y['Pressure (millibars)'] == 0, 'Pressure (millibars)'] = np.nan

# Encode categorical features
print("\nEncoding categorical features...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    
    # Find and replace 'nan' codes with actual NaN
    try:
        nan_code = list(le.classes_).index('nan')
        X[col] = X[col].replace(nan_code, np.nan)
    except ValueError:
        pass

# Combine X and y for joint imputation
print("\nImputing missing values with KNNImputer...")
combined_data = pd.concat([X, y], axis=1)
imputer = KNNImputer(n_neighbors=5)
combined_imputed = imputer.fit_transform(combined_data)
combined_df = pd.DataFrame(combined_imputed, columns=combined_data.columns)

# Separate back to X and y
X = combined_df.drop(target_columns, axis=1)
y = combined_df[target_columns]

print(f"Missing values after imputation:")
print(f"  Features: {X.isnull().sum().sum()}")
print(f"  Targets: {y.isnull().sum().sum()}")
print("Preprocessing complete.")

# --- 2. EXPLORATORY DATA ANALYSIS ---
print("\n2. EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# Target correlation
target_corr = y.corr()
print("\nTarget correlation:")
print(target_corr)

# Visualize target distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y['Pressure (millibars)'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Pressure (millibars)', fontweight='bold')
axes[0].set_ylabel('Frequency', fontweight='bold')
axes[0].set_title('Distribution of Pressure', fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].hist(y['Humidity'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
axes[1].set_xlabel('Humidity', fontweight='bold')
axes[1].set_ylabel('Frequency', fontweight='bold')
axes[1].set_title('Distribution of Humidity', fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('multi_output_results/target_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Target distributions plot saved")
plt.close()

# Scatter plot of targets
plt.figure(figsize=(8, 6))
plt.scatter(y['Pressure (millibars)'], y['Humidity'], alpha=0.3, s=10)
plt.xlabel('Pressure (millibars)', fontweight='bold')
plt.ylabel('Humidity', fontweight='bold')
plt.title(f'Pressure vs Humidity\nCorrelation: {target_corr.iloc[0, 1]:.4f}', fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('multi_output_results/pressure_humidity_scatter.png', dpi=300, bbox_inches='tight')
print("✓ Pressure-Humidity scatter plot saved")
plt.close()

# --- 3. DATA SPLITTING ---
print("\n3. DATA SPLITTING")
print("=" * 70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# --- 4. MODEL TRAINING - BASELINE XGBOOST ---
print("\n4. BASELINE XGBOOST MULTI-OUTPUT MODEL")
print("=" * 70)

# Create baseline XGBoost multi-output model
print("\nTraining baseline XGBoost MultiOutputRegressor...")
start_time = time.time()

baseline_model = MultiOutputRegressor(
    xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
)

baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

baseline_time = time.time() - start_time

# Evaluate baseline
r2_pressure_base = r2_score(y_test['Pressure (millibars)'], y_pred_baseline[:, 0])
r2_humidity_base = r2_score(y_test['Humidity'], y_pred_baseline[:, 1])
mse_pressure_base = mean_squared_error(y_test['Pressure (millibars)'], y_pred_baseline[:, 0])
mse_humidity_base = mean_squared_error(y_test['Humidity'], y_pred_baseline[:, 1])
mae_pressure_base = mean_absolute_error(y_test['Pressure (millibars)'], y_pred_baseline[:, 0])
mae_humidity_base = mean_absolute_error(y_test['Humidity'], y_pred_baseline[:, 1])

print(f"\nBaseline Results (Training time: {baseline_time:.2f}s):")
print(f"\nPressure:")
print(f"  R² Score: {r2_pressure_base:.4f}")
print(f"  MSE: {mse_pressure_base:.4f}")
print(f"  MAE: {mae_pressure_base:.4f}")
print(f"\nHumidity:")
print(f"  R² Score: {r2_humidity_base:.4f}")
print(f"  MSE: {mse_humidity_base:.4f}")
print(f"  MAE: {mae_humidity_base:.4f}")

# --- 5. NORMALIZATION COMPARISON ---
print("\n5. NORMALIZATION COMPARISON")
print("=" * 70)

scalers = {
    'No Scaling': None,
    'StandardScaler': StandardScaler()
}

normalization_results = []

for scaler_name, scaler in scalers.items():
    print(f"\nTesting {scaler_name}...")
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    
    model = MultiOutputRegressor(
        xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    )
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2_pressure = r2_score(y_test['Pressure (millibars)'], y_pred[:, 0])
    r2_humidity = r2_score(y_test['Humidity'], y_pred[:, 1])
    mse_pressure = mean_squared_error(y_test['Pressure (millibars)'], y_pred[:, 0])
    mse_humidity = mean_squared_error(y_test['Humidity'], y_pred[:, 1])
    
    avg_r2 = (r2_pressure + r2_humidity) / 2
    
    normalization_results.append({
        'Scaler': scaler_name,
        'R²_Pressure': r2_pressure,
        'R²_Humidity': r2_humidity,
        'Avg_R²': avg_r2,
        'MSE_Pressure': mse_pressure,
        'MSE_Humidity': mse_humidity
    })
    
    print(f"  Pressure R²: {r2_pressure:.4f}, Humidity R²: {r2_humidity:.4f}, Avg R²: {avg_r2:.4f}")

norm_df = pd.DataFrame(normalization_results)
best_scaler_name = norm_df.loc[norm_df['Avg_R²'].idxmax(), 'Scaler']
print(f"\n✓ Best normalization: {best_scaler_name}")

# --- 6. GRIDSEARCH HYPERPARAMETER TUNING ---
print("\n6. GRIDSEARCH HYPERPARAMETER TUNING")
print("=" * 70)

# Prepare data with best scaler
if best_scaler_name != 'No Scaling':
    print(f"\nApplying {best_scaler_name}...")
    best_scaler = StandardScaler()
    X_train_final = best_scaler.fit_transform(X_train)
    X_test_final = best_scaler.transform(X_test)
else:
    best_scaler = None
    X_train_final = X_train.values
    X_test_final = X_test.values

# Define parameter grid
param_grid = {
    'estimator__n_estimators': [100, 150, 200],
    'estimator__max_depth': [5, 7, 9],
    'estimator__learning_rate': [0.05, 0.1, 0.15],
    'estimator__min_child_weight': [1, 3],
    'estimator__subsample': [0.8, 0.9],
    'estimator__colsample_bytree': [0.8, 0.9]
}

print(f"\nParameter grid:")
for param, values in param_grid.items():
    print(f"  {param}: {values}")

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"\nTotal combinations: {total_combinations}")
print(f"Total fits (with 5-fold CV): {total_combinations * 5}")

print("\nStarting GridSearchCV (this may take several minutes)...")
start_time = time.time()

grid_search = GridSearchCV(
    MultiOutputRegressor(xgb.XGBRegressor(random_state=42, n_jobs=-1)),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_final, y_train)
grid_time = time.time() - start_time

print(f"\n✓ GridSearch completed in {grid_time:.2f}s ({grid_time/60:.2f} minutes)")

# Best parameters
print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

# --- 7. EVALUATE BEST MODEL ---
print("\n7. EVALUATING BEST MODEL")
print("=" * 70)

best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_final)

# Calculate metrics for both targets
r2_pressure = r2_score(y_test['Pressure (millibars)'], y_pred_best[:, 0])
r2_humidity = r2_score(y_test['Humidity'], y_pred_best[:, 1])
mse_pressure = mean_squared_error(y_test['Pressure (millibars)'], y_pred_best[:, 0])
mse_humidity = mean_squared_error(y_test['Humidity'], y_pred_best[:, 1])
mae_pressure = mean_absolute_error(y_test['Pressure (millibars)'], y_pred_best[:, 0])
mae_humidity = mean_absolute_error(y_test['Humidity'], y_pred_best[:, 1])

avg_r2 = (r2_pressure + r2_humidity) / 2
avg_mse = (mse_pressure + mse_humidity) / 2
avg_mae = (mae_pressure + mae_humidity) / 2

print(f"\nBest Model Performance:")
print(f"\nPressure (millibars):")
print(f"  R² Score: {r2_pressure:.4f}")
print(f"  MSE: {mse_pressure:.4f}")
print(f"  MAE: {mae_pressure:.4f}")

print(f"\nHumidity:")
print(f"  R² Score: {r2_humidity:.4f}")
print(f"  MSE: {mse_humidity:.4f}")
print(f"  MAE: {mae_humidity:.4f}")

print(f"\nAverage Metrics:")
print(f"  Avg R²: {avg_r2:.4f}")
print(f"  Avg MSE: {avg_mse:.4f}")
print(f"  Avg MAE: {avg_mae:.4f}")

# Improvement over baseline
print(f"\nImprovement over baseline:")
print(f"  Pressure R²: {(r2_pressure - r2_pressure_base)*100:+.2f}%")
print(f"  Humidity R²: {(r2_humidity - r2_humidity_base)*100:+.2f}%")

# --- 8. SAVE BEST MODEL ---
print("\n8. SAVING BEST MODEL FOR PRODUCTION")
print("=" * 70)

# Save the model
model_path = 'multi_output_results/models/best_model.joblib'
dump(best_model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save the scaler if used
if best_scaler is not None:
    scaler_path = 'multi_output_results/models/scaler.joblib'
    dump(best_scaler, scaler_path)
    print(f"✓ Scaler saved to: {scaler_path}")

# Create metadata
metadata = {
    'model_name': 'XGBoost_MultiOutput_GridSearch',
    'model_type': 'MultiOutputRegressor(XGBRegressor)',
    'target_variables': target_columns,
    'normalization': best_scaler_name,
    'best_parameters': {k.replace('estimator__', ''): v for k, v in grid_search.best_params_.items()},
    'performance_metrics': {
        'pressure': {
            'r2_score': float(r2_pressure),
            'mse': float(mse_pressure),
            'mae': float(mae_pressure)
        },
        'humidity': {
            'r2_score': float(r2_humidity),
            'mse': float(mse_humidity),
            'mae': float(mae_humidity)
        },
        'average': {
            'r2_score': float(avg_r2),
            'mse': float(avg_mse),
            'mae': float(avg_mae)
        }
    },
    'training_details': {
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'gridsearch_time_seconds': float(grid_time),
        'total_cv_fits': int(total_combinations * 5)
    },
    'feature_names': list(X.columns),
    'preprocessing': {
        'imputation': 'KNNImputer (n_neighbors=5)',
        'dropped_columns': columns_to_drop,
        'zero_pressure_handled': bool(zero_pressure_count > 0),
        'zero_pressure_count': int(zero_pressure_count)
    }
}

# Save metadata
metadata_path = 'multi_output_results/models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("MODEL PERSISTENCE COMPLETE")
print("="*70)
print("\nSaved files:")
print(f"  1. {model_path}")
if best_scaler is not None:
    print(f"  2. {scaler_path}")
print(f"  3. {metadata_path}")

# --- 9. VISUALIZATIONS ---
print("\n9. GENERATING VISUALIZATIONS")
print("=" * 70)

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. Pressure: True vs Predicted
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(y_test['Pressure (millibars)'], y_pred_best[:, 0], alpha=0.3, s=10)
min_p = min(y_test['Pressure (millibars)'].min(), y_pred_best[:, 0].min())
max_p = max(y_test['Pressure (millibars)'].max(), y_pred_best[:, 0].max())
ax1.plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('True Pressure (millibars)', fontweight='bold')
ax1.set_ylabel('Predicted Pressure (millibars)', fontweight='bold')
ax1.set_title(f'Pressure Predictions\nR²={r2_pressure:.4f}', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Humidity: True vs Predicted
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(y_test['Humidity'], y_pred_best[:, 1], alpha=0.3, s=10)
min_h = min(y_test['Humidity'].min(), y_pred_best[:, 1].min())
max_h = max(y_test['Humidity'].max(), y_pred_best[:, 1].max())
ax2.plot([min_h, max_h], [min_h, max_h], 'r--', linewidth=2, label='Perfect Prediction')
ax2.set_xlabel('True Humidity', fontweight='bold')
ax2.set_ylabel('Predicted Humidity', fontweight='bold')
ax2.set_title(f'Humidity Predictions\nR²={r2_humidity:.4f}', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Performance Metrics Comparison
ax3 = fig.add_subplot(gs[0, 2])
metrics = ['R² Score', 'MSE', 'MAE']
pressure_vals = [r2_pressure, mse_pressure, mae_pressure]
humidity_vals = [r2_humidity, mse_humidity, mae_humidity]
x_pos = np.arange(len(metrics))
width = 0.35
ax3.bar(x_pos - width/2, pressure_vals, width, label='Pressure', alpha=0.8)
ax3.bar(x_pos + width/2, humidity_vals, width, label='Humidity', alpha=0.8)
ax3.set_ylabel('Score', fontweight='bold')
ax3.set_title('Performance Metrics Comparison', fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(metrics)
ax3.legend()
ax3.grid(alpha=0.3, axis='y')

# 4. Pressure Residuals
ax4 = fig.add_subplot(gs[1, 0])
pressure_residuals = y_test['Pressure (millibars)'].values - y_pred_best[:, 0]
ax4.scatter(y_pred_best[:, 0], pressure_residuals, alpha=0.3, s=10)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Predicted Pressure (millibars)', fontweight='bold')
ax4.set_ylabel('Residuals', fontweight='bold')
ax4.set_title('Pressure Residuals', fontweight='bold')
ax4.grid(alpha=0.3)

# 5. Humidity Residuals
ax5 = fig.add_subplot(gs[1, 1])
humidity_residuals = y_test['Humidity'].values - y_pred_best[:, 1]
ax5.scatter(y_pred_best[:, 1], humidity_residuals, alpha=0.3, s=10)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Humidity', fontweight='bold')
ax5.set_ylabel('Residuals', fontweight='bold')
ax5.set_title('Humidity Residuals', fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Residual Distributions
ax6 = fig.add_subplot(gs[1, 2])
ax6.hist(pressure_residuals, bins=50, alpha=0.6, label='Pressure', edgecolor='black')
ax6.hist(humidity_residuals, bins=50, alpha=0.6, label='Humidity', edgecolor='black')
ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax6.set_xlabel('Residuals', fontweight='bold')
ax6.set_ylabel('Frequency', fontweight='bold')
ax6.set_title('Residual Distributions', fontweight='bold')
ax6.legend()
ax6.grid(alpha=0.3, axis='y')

# 7. Pressure Prediction Timeline (first 500 samples)
ax7 = fig.add_subplot(gs[2, 0])
sample_size = min(500, len(y_test))
sorted_idx = np.argsort(y_test['Pressure (millibars)'].values)[:sample_size]
ax7.plot(range(sample_size), y_test['Pressure (millibars)'].values[sorted_idx], 
         'o-', label='True', alpha=0.6, markersize=3, linewidth=1)
ax7.plot(range(sample_size), y_pred_best[sorted_idx, 0], 
         's-', label='Predicted', alpha=0.6, markersize=3, linewidth=1)
ax7.set_xlabel('Sample Index (sorted)', fontweight='bold')
ax7.set_ylabel('Pressure (millibars)', fontweight='bold')
ax7.set_title(f'Pressure Timeline (First {sample_size} Samples)', fontweight='bold')
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Humidity Prediction Timeline (first 500 samples)
ax8 = fig.add_subplot(gs[2, 1])
sorted_idx_h = np.argsort(y_test['Humidity'].values)[:sample_size]
ax8.plot(range(sample_size), y_test['Humidity'].values[sorted_idx_h], 
         'o-', label='True', alpha=0.6, markersize=3, linewidth=1)
ax8.plot(range(sample_size), y_pred_best[sorted_idx_h, 1], 
         's-', label='Predicted', alpha=0.6, markersize=3, linewidth=1)
ax8.set_xlabel('Sample Index (sorted)', fontweight='bold')
ax8.set_ylabel('Humidity', fontweight='bold')
ax8.set_title(f'Humidity Timeline (First {sample_size} Samples)', fontweight='bold')
ax8.legend()
ax8.grid(alpha=0.3)

# 9. Feature Importance (from first estimator as representative)
ax9 = fig.add_subplot(gs[2, 2])
feature_importance = best_model.estimators_[0].feature_importances_
top_n = 15
top_indices = np.argsort(feature_importance)[-top_n:]
ax9.barh(range(top_n), feature_importance[top_indices])
ax9.set_yticks(range(top_n))
ax9.set_yticklabels([X.columns[i] for i in top_indices], fontsize=8)
ax9.set_xlabel('Importance', fontweight='bold')
ax9.set_title(f'Top {top_n} Feature Importances', fontweight='bold')
ax9.grid(alpha=0.3, axis='x')

plt.suptitle('XGBoost Multi-Output Regression: Pressure & Humidity Prediction Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.savefig('multi_output_results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Comprehensive analysis plot saved")
plt.close()

# Save results summary
results_summary = {
    'Model': 'XGBoost MultiOutput',
    'Normalization': best_scaler_name,
    'Pressure_R2': r2_pressure,
    'Pressure_MSE': mse_pressure,
    'Pressure_MAE': mae_pressure,
    'Humidity_R2': r2_humidity,
    'Humidity_MSE': mse_humidity,
    'Humidity_MAE': mae_humidity,
    'Average_R2': avg_r2,
    'GridSearch_Time_Minutes': grid_time / 60
}

results_df = pd.DataFrame([results_summary])
results_df.to_csv('multi_output_results/model_performance_summary.csv', index=False)
print("✓ Performance summary saved")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print("- multi_output_results/target_distributions.png")
print("- multi_output_results/pressure_humidity_scatter.png")
print("- multi_output_results/comprehensive_analysis.png")
print("- multi_output_results/model_performance_summary.csv")
print("- multi_output_results/models/best_model.joblib")
if best_scaler is not None:
    print("- multi_output_results/models/scaler.joblib")
print("- multi_output_results/models/model_metadata.json")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"✓ Successfully trained XGBoost multi-output model")
print(f"✓ Predicting: Pressure (millibars) & Humidity")
print(f"✓ Best normalization: {best_scaler_name}")
print(f"✓ Average R² Score: {avg_r2:.4f}")
print(f"✓ Pressure R²: {r2_pressure:.4f} | Humidity R²: {r2_humidity:.4f}")
print(f"✓ Model saved and ready for production use")
print(f"{'='*70}")
