import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== TEMPERATURE REGRESSION ANALYSIS PIPELINE ===\n")

# Create a directory for results if it doesn't exist
if not os.path.exists('regression_results'):
    os.makedirs('regression_results')

# --- 1. LOAD AND PREPROCESS DATA ---
print("1. LOADING AND PREPROCESSING DATA")
print("=" * 50)

# Load the dataset
df = pd.read_csv('Dataset1.csv')
print(f"Original dataset shape: {df.shape}")

# Drop specified columns
columns_to_drop = ['Apparent Temperature (C)', 'Formatted Date', 'Daily Summary']
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")
print(f"Dataset shape after dropping columns: {df.shape}")

# Separate features and target
X = df.drop('Temperature (C)', axis=1)
y = df['Temperature (C)']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"\nCategorical features: {list(categorical_features)}")
print(f"Numerical features: {list(numerical_features)}")

# Encode categorical features
print("\nEncoding categorical features...")
for col in categorical_features:
    le = LabelEncoder()
    # Use -1 for missing values so KNNImputer can find them
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    # Find the code for 'nan' string which represents original NaNs
    nan_code = -1
    try:
        nan_code = list(le.classes_).index('nan')
    except ValueError:
        pass # No 'nan' string found, so no original NaNs
    
    if nan_code != -1:
        X[col] = X[col].replace(nan_code, np.nan)

# Impute missing values using KNNImputer
print("\nImputing missing values with KNNImputer...")
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Total missing values after imputation: {X.isnull().sum().sum()}")
print("Preprocessing complete.")

# --- 2. MODEL TUNING AND COMPARISON ---
print("\n2. MODEL TUNING AND COMPARISON")
print("=" * 50)

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Define models to test
models_to_test = {
    'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

final_results = []

# Main loop to iterate through each model
for model_name, base_model in models_to_test.items():
    print(f"\n{'='*60}")
    print(f"TUNING MODEL: {model_name}")
    print(f"{'='*60}")
    
    # --- 2.1 TUNE N_ESTIMATORS ---
    print(f"\n--- Step 1: Tuning n_estimators for {model_name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    n_estimators_list = range(50, 251, 25)
    r2_scores_estimators = []

    for n in n_estimators_list:
        model = base_model.set_params(n_estimators=n)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores_estimators.append(r2)
        print(f"  n_estimators={n}, R²={r2:.4f}")

    best_n_estimators = n_estimators_list[np.argmax(r2_scores_estimators)]
    best_n_r2 = max(r2_scores_estimators)
    print(f"✓ Best n_estimators for {model_name}: {best_n_estimators} (R²={best_n_r2:.4f})")
    
    # Save individual model n_estimators tuning
    estimators_df = pd.DataFrame({'n_estimators': list(n_estimators_list), 'r2_score': r2_scores_estimators})
    estimators_df.to_csv(f'regression_results/{model_name.lower()}_n_estimators_tuning.csv', index=False)

    estimators_df.to_csv(f'regression_results/{model_name.lower()}_n_estimators_tuning.csv', index=False)

    # --- 2.2 TUNING TRAINING DATA PERCENTAGE ---
    print(f"\n--- Step 2: Tuning data percentage for {model_name} ---")
    percentages = np.arange(0.1, 1.1, 0.1)
    r2_scores_pct = []

    for pct in percentages:
        print(f"  Training with {pct*100:.0f}% of data...", end='')
        
        if np.isclose(pct, 1.0):
            X_sample, y_sample = X, y
        else:
            X_sample, _, y_sample, _ = train_test_split(X, y, train_size=pct, random_state=42)
        
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        
        model = base_model.set_params(n_estimators=best_n_estimators)
        model.fit(X_train_s, y_train_s)
        y_pred_s = model.predict(X_test_s)
        r2 = r2_score(y_test_s, y_pred_s)
        r2_scores_pct.append(r2)
        print(f" R²={r2:.4f}")

    best_data_pct = percentages[np.argmax(r2_scores_pct)]
    best_pct_r2 = max(r2_scores_pct)
    print(f"✓ Best data percentage for {model_name}: {best_data_pct*100:.0f}% (R²={best_pct_r2:.4f})")
    
    # Save individual model data percentage tuning
    percentage_df = pd.DataFrame({'data_percentage': [f"{p*100:.0f}%" for p in percentages], 'r2_score': r2_scores_pct})
    percentage_df.to_csv(f'regression_results/{model_name.lower()}_data_percentage_tuning.csv', index=False)

    # --- 2.3 ANALYZING SCALER IMPACT ---
    print(f"\n--- Step 3: Analyzing scaler impact for {model_name} ---")
    
    if best_data_pct < 1.0:
        X_final_sample, _, y_final_sample, _ = train_test_split(X, y, train_size=best_data_pct, random_state=42)
    else:
        X_final_sample, y_final_sample = X, y
        
    X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_final_sample, y_final_sample, test_size=0.2, random_state=42)

    scalers = {
        'No Scaling': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler()
    }
    
    scaler_results = []

    for scaler_name, scaler_instance in scalers.items():
        X_train_scaled, X_test_scaled = X_train_fs.copy(), X_test_fs.copy()
        
        if scaler_instance:
            X_train_scaled[numerical_features] = scaler_instance.fit_transform(X_train_fs[numerical_features])
            X_test_scaled[numerical_features] = scaler_instance.transform(X_test_fs[numerical_features])
        
        model = base_model.set_params(n_estimators=best_n_estimators)
        model.fit(X_train_scaled, y_train_fs)
        y_pred_final = model.predict(X_test_scaled)
        
        r2 = r2_score(y_test_fs, y_pred_final)
        mse = mean_squared_error(y_test_fs, y_pred_final)
        ev = explained_variance_score(y_test_fs, y_pred_final)
        
        scaler_results.append({'scaler': scaler_name, 'r2_score': r2, 'mse': mse, 'explained_variance': ev})
        print(f"  {scaler_name}: R²={r2:.4f}, MSE={mse:.4f}")

    # Find best scaler for the current model
    best_scaler_result = max(scaler_results, key=lambda x: x['r2_score'])
    print(f"✓ Best scaler for {model_name}: {best_scaler_result['scaler']} (R²={best_scaler_result['r2_score']:.4f})")
    
    # Get predictions for visualization (using best scaler configuration)
    X_train_viz, X_test_viz = X_train_fs.copy(), X_test_fs.copy()
    if best_scaler_result['scaler'] != 'No Scaling':
        scaler_viz = scalers[best_scaler_result['scaler']]
        X_train_viz[numerical_features] = scaler_viz.fit_transform(X_train_fs[numerical_features])
        X_test_viz[numerical_features] = scaler_viz.transform(X_test_fs[numerical_features])
    
    model_viz = base_model.set_params(n_estimators=best_n_estimators)
    model_viz.fit(X_train_viz, y_train_fs)
    y_pred_viz = model_viz.predict(X_test_viz)
    
    # Calculate MAE for reporting
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(y_test_fs, y_pred_viz)
    
    # Append the best result for the current model to the final list
    final_results.append({
        'model': model_name,
        'n_estimators': best_n_estimators,
        'data_percentage': best_data_pct * 100,
        'scaler': best_scaler_result['scaler'],
        'r2_score': best_scaler_result['r2_score'],
        'mse': best_scaler_result['mse'],
        'mae': mae,
        'explained_variance': best_scaler_result['explained_variance'],
        'y_test': y_test_fs,
        'y_pred': y_pred_viz
    })


# --- 3. LOG AND PLOT FINAL RESULTS ---
print(f"\n\n{'='*60}")
print("FINAL RESULTS - ALL MODELS OPTIMIZED")
print(f"{'='*60}")

# Extract y_test and y_pred for visualizations before creating DataFrame
model_predictions = {result['model']: {'y_test': result['y_test'], 'y_pred': result['y_pred']} 
                     for result in final_results}

# Create DataFrame from the best results of each model (excluding y_test and y_pred)
model_comparison_df = pd.DataFrame([{k: v for k, v in result.items() if k not in ['y_test', 'y_pred']} 
                                    for result in final_results])
model_comparison_df = model_comparison_df.sort_values('r2_score', ascending=False)

# Save results to CSV
csv_path = 'regression_results/model_comparison_results.csv'
model_comparison_df.to_csv(csv_path, index=False)
print(f"\n✓ Final model comparison results saved to '{csv_path}'")
print("\nFinal Results Summary:")
print(model_comparison_df.to_string())

# --- VISUALIZATION 1: Model Performance Comparison ---
fig, axes = plt.subplots(1, 4, figsize=(28, 6))
fig.suptitle('Final Model Performance Comparison (Each Model Individually Optimized)', fontsize=16, fontweight='bold')

# R-squared
sns.barplot(x='r2_score', y='model', data=model_comparison_df, ax=axes[0], orient='h', palette='viridis')
axes[0].set_title('R-squared (R²)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('R² Score', fontsize=12)
axes[0].set_ylabel('Model', fontsize=12)
axes[0].grid(axis='x', alpha=0.3)
for i, v in enumerate(model_comparison_df['r2_score']):
    axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

# Mean Squared Error
sns.barplot(x='mse', y='model', data=model_comparison_df, ax=axes[1], orient='h', palette='rocket')
axes[1].set_title('Mean Squared Error (MSE)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('MSE', fontsize=12)
axes[1].set_ylabel('', fontsize=12)
axes[1].grid(axis='x', alpha=0.3)
for i, v in enumerate(model_comparison_df['mse']):
    axes[1].text(v + 0.5, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

# Mean Absolute Error
sns.barplot(x='mae', y='model', data=model_comparison_df, ax=axes[2], orient='h', palette='mako')
axes[2].set_title('Mean Absolute Error (MAE)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('MAE', fontsize=12)
axes[2].set_ylabel('', fontsize=12)
axes[2].grid(axis='x', alpha=0.3)
for i, v in enumerate(model_comparison_df['mae']):
    axes[2].text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=10, fontweight='bold')

# Explained Variance
sns.barplot(x='explained_variance', y='model', data=model_comparison_df, ax=axes[3], orient='h', palette='crest')
axes[3].set_title('Explained Variance Score', fontsize=14, fontweight='bold')
axes[3].set_xlabel('Explained Variance', fontsize=12)
axes[3].set_ylabel('', fontsize=12)
axes[3].grid(axis='x', alpha=0.3)
for i, v in enumerate(model_comparison_df['explained_variance']):
    axes[3].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=10, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('regression_results/final_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison plot saved to 'regression_results/final_model_comparison.png'")
plt.close()

# --- VISUALIZATION 2: True vs Predicted Values for Each Model ---
print("\nGenerating true vs predicted visualizations for each model...")

for model_name in model_comparison_df['model']:
    y_test_model = model_predictions[model_name]['y_test']
    y_pred_model = model_predictions[model_name]['y_pred']
    
    model_info = model_comparison_df[model_comparison_df['model'] == model_name].iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'{model_name} - Prediction Analysis\n'
                 f'R²={model_info["r2_score"]:.4f}, MAE={model_info["mae"]:.2f}°C, MSE={model_info["mse"]:.2f}',
                 fontsize=16, fontweight='bold')
    
    # 1. Scatter Plot: True vs Predicted
    ax1 = axes[0, 0]
    ax1.scatter(y_test_model, y_pred_model, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    
    # Add perfect prediction line
    min_val = min(y_test_model.min(), y_pred_model.min())
    max_val = max(y_test_model.max(), y_pred_model.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('True Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
    ax1.set_title('True vs Predicted Values', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(alpha=0.3)
    
    # Add correlation text
    correlation = np.corrcoef(y_test_model, y_pred_model)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. Residual Plot
    ax2 = axes[0, 1]
    residuals = y_test_model - y_pred_model
    ax2.scatter(y_pred_model, residuals, alpha=0.5, s=30, edgecolors='k', linewidths=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Residuals (True - Predicted)', fontsize=12, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # Add residual statistics
    residual_mean = residuals.mean()
    residual_std = residuals.std()
    ax2.text(0.05, 0.95, f'Mean: {residual_mean:.3f}°C\nStd: {residual_std:.3f}°C', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 3. Distribution of Residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax3.set_xlabel('Residuals (°C)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Distribution of Prediction Errors', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(alpha=0.3, axis='y')
    
    # 4. Prediction Timeline (first 500 samples for clarity)
    ax4 = axes[1, 1]
    sample_size = min(500, len(y_test_model))
    indices = np.arange(sample_size)
    
    # Sort by true values for better visualization
    sorted_indices = np.argsort(y_test_model.values)[:sample_size]
    
    ax4.plot(indices, y_test_model.values[sorted_indices], 'o-', label='True Values', 
             alpha=0.6, markersize=4, linewidth=1.5)
    ax4.plot(indices, y_pred_model[sorted_indices], 's-', label='Predicted Values', 
             alpha=0.6, markersize=4, linewidth=1.5)
    ax4.set_xlabel('Sample Index (sorted by true value)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Temperature (°C)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Prediction Comparison (First {sample_size} Samples)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'regression_results/{model_name.lower()}_prediction_analysis.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ {model_name} prediction analysis saved")
    plt.close()

# --- VISUALIZATION 3: Combined Comparison of All Models ---
print("\nGenerating combined comparison plot...")

fig, axes = plt.subplots(1, len(model_predictions), figsize=(8 * len(model_predictions), 7))
if len(model_predictions) == 1:
    axes = [axes]

fig.suptitle('True vs Predicted - All Models Comparison', fontsize=16, fontweight='bold')

for idx, (model_name, predictions) in enumerate(model_predictions.items()):
    y_test_model = predictions['y_test']
    y_pred_model = predictions['y_pred']
    model_info = model_comparison_df[model_comparison_df['model'] == model_name].iloc[0]
    
    ax = axes[idx]
    ax.scatter(y_test_model, y_pred_model, alpha=0.4, s=20, edgecolors='k', linewidths=0.3)
    
    # Perfect prediction line
    min_val = min(y_test_model.min(), y_pred_model.min())
    max_val = max(y_test_model.max(), y_pred_model.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
    ax.set_title(f'{model_name}\nR²={model_info["r2_score"]:.4f}, MAE={model_info["mae"]:.2f}°C', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('regression_results/all_models_prediction_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Combined comparison plot saved to 'regression_results/all_models_prediction_comparison.png'")
plt.close()

print("\n=== ANALYSIS COMPLETE ===")
print("Generated files:")
for model_name in models_to_test.keys():
    print(f"- regression_results/{model_name.lower()}_n_estimators_tuning.csv")
    print(f"- regression_results/{model_name.lower()}_data_percentage_tuning.csv")
    print(f"- regression_results/{model_name.lower()}_prediction_analysis.png")
print("- regression_results/model_comparison_results.csv")
print("- regression_results/final_model_comparison.png")
print("- regression_results/all_models_prediction_comparison.png")
