import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("ADVANCED MODEL COMPARISON: SVM Kernels & XGBoost with Shuffling")
print("="*80)

# Create directory for results
if not os.path.exists('advanced_results'):
    os.makedirs('advanced_results')

# --- 1. LOAD AND PREPROCESS DATA ---
print("\n1. Loading and preprocessing data...")
df = pd.read_csv('Dataset1.csv')
columns_to_drop = ['Apparent Temperature (C)', 'Formatted Date', 'Daily Summary']
df = df.drop(columns=columns_to_drop)

X = df.drop('Temperature (C)', axis=1)
y = df['Temperature (C)']

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Encode categorical features
for col in categorical_features:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    nan_code = -1
    try:
        nan_code = list(le.classes_).index('nan')
    except ValueError:
        pass
    if nan_code != -1:
        X[col] = X[col].replace(nan_code, np.nan)

# Impute missing values
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"✓ Dataset shape: {X.shape}")
print(f"✓ Data preprocessing complete\n")

# --- 2. SVM WITH DIFFERENT KERNELS ---
print("="*80)
print("2. TESTING SVM WITH DIFFERENT KERNELS")
print("="*80)

# Use a sample of data for SVM (faster training)
sample_size = 0.1  # Use 10% of data for SVM
print(f"\nUsing {sample_size*100:.0f}% of data for SVM experiments (faster training)")
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=sample_size, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

# Scale data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape[0]} samples")
print(f"Test set: {X_test_scaled.shape[0]} samples\n")

# Define SVM kernels to test
svm_kernels = ['linear', 'poly', 'rbf', 'sigmoid']
svm_results = []

for kernel in svm_kernels:
    print(f"\nTraining SVM with '{kernel}' kernel...", end=' ')
    
    # Configure SVM based on kernel
    if kernel == 'poly':
        model = SVR(kernel=kernel, degree=3, C=1.0, epsilon=0.1)
    else:
        model = SVR(kernel=kernel, C=1.0, epsilon=0.1)
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    
    svm_results.append({
        'kernel': kernel,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'y_test': y_test,
        'y_pred': y_pred_test
    })
    
    print(f"Test R²={test_r2:.4f}, MAE={test_mae:.2f}°C")

svm_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred']} 
                        for r in svm_results])
svm_df.to_csv('advanced_results/svm_kernels_comparison.csv', index=False)
print("\n✓ SVM results saved to 'advanced_results/svm_kernels_comparison.csv'")

# --- 3. XGBOOST WITH DATA SHUFFLING ---
print("\n" + "="*80)
print("3. TESTING XGBOOST WITH DIFFERENT DATA SHUFFLING")
print("="*80)

# Use larger sample for XGBoost
xgb_sample_size = 0.3  # Use 30% of data
print(f"\nUsing {xgb_sample_size*100:.0f}% of data for XGBoost experiments")
X_xgb, _, y_xgb, _ = train_test_split(X, y, train_size=xgb_sample_size, random_state=42)

# Test different random states (shuffling)
random_states = [42, 123, 456, 789, 999]
xgb_results = []

for rs in random_states:
    print(f"\nTraining XGBoost with shuffle seed {rs}...", end=' ')
    
    # Split with different random state
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
        X_xgb, y_xgb, test_size=0.2, random_state=rs, shuffle=True
    )
    
    # Train XGBoost
    model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_xgb, y_train_xgb)
    
    # Predictions
    y_pred_train = model.predict(X_train_xgb)
    y_pred_test = model.predict(X_test_xgb)
    
    # Metrics
    train_r2 = r2_score(y_train_xgb, y_pred_train)
    test_r2 = r2_score(y_test_xgb, y_pred_test)
    train_mae = mean_absolute_error(y_train_xgb, y_pred_train)
    test_mae = mean_absolute_error(y_test_xgb, y_pred_test)
    train_mse = mean_squared_error(y_train_xgb, y_pred_train)
    test_mse = mean_squared_error(y_test_xgb, y_pred_test)
    
    xgb_results.append({
        'random_state': rs,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'y_test': y_test_xgb,
        'y_pred': y_pred_test
    })
    
    print(f"Test R²={test_r2:.4f}, MAE={test_mae:.2f}°C")

xgb_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['y_test', 'y_pred']} 
                        for r in xgb_results])
xgb_df.to_csv('advanced_results/xgboost_shuffling_comparison.csv', index=False)
print("\n✓ XGBoost shuffling results saved to 'advanced_results/xgboost_shuffling_comparison.csv'")

# --- 4. VISUALIZATIONS ---
print("\n" + "="*80)
print("4. GENERATING VISUALIZATIONS")
print("="*80)

# PLOT 1: SVM Kernels Comparison
print("\n  Generating SVM kernels comparison plots...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('SVM Performance with Different Kernels', fontsize=18, fontweight='bold')

# 1. R² Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(svm_df))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, svm_df['train_r2'], width, label='Train R²', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x_pos + width/2, svm_df['test_r2'], width, label='Test R²', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Kernel', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Score by Kernel', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(svm_df['kernel'])
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3, axis='y')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
            ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
            ha='center', va='bottom', fontsize=9)

# 2. MAE Comparison
ax2 = axes[0, 1]
bars3 = ax2.bar(x_pos - width/2, svm_df['train_mae'], width, label='Train MAE', alpha=0.8, edgecolor='black')
bars4 = ax2.bar(x_pos + width/2, svm_df['test_mae'], width, label='Test MAE', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Kernel', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Mean Absolute Error by Kernel', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(svm_df['kernel'])
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3, axis='y')
for bar in bars3:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=9)
for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=9)

# 3. Test R² Line Plot
ax3 = axes[0, 2]
ax3.plot(svm_df['kernel'], svm_df['test_r2'], 'o-', linewidth=2, markersize=10, color='darkblue')
ax3.set_xlabel('Kernel', fontsize=12, fontweight='bold')
ax3.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
ax3.set_title('Test R² Across Kernels', fontsize=13, fontweight='bold')
ax3.grid(alpha=0.3)
for i, (kernel, r2) in enumerate(zip(svm_df['kernel'], svm_df['test_r2'])):
    ax3.text(i, r2 + 0.01, f'{r2:.4f}', ha='center', fontsize=10, fontweight='bold')

# 4. True vs Predicted for best kernel
ax4 = axes[1, 0]
best_svm_idx = svm_df['test_r2'].idxmax()
best_svm = svm_results[best_svm_idx]
y_test_svm = best_svm['y_test']
y_pred_svm = best_svm['y_pred']
ax4.scatter(y_test_svm, y_pred_svm, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
min_val = min(y_test_svm.min(), y_pred_svm.min())
max_val = max(y_test_svm.max(), y_pred_svm.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('True Temperature (°C)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
ax4.set_title(f'Best Kernel: {best_svm["kernel"]} (R²={best_svm["test_r2"]:.4f})', 
              fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# 5. Residuals for best kernel
ax5 = axes[1, 1]
residuals_svm = y_test_svm.values - y_pred_svm
ax5.scatter(y_pred_svm, residuals_svm, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Residuals (°C)', fontsize=12, fontweight='bold')
ax5.set_title(f'Residual Plot - {best_svm["kernel"]} kernel', fontsize=13, fontweight='bold')
ax5.grid(alpha=0.3)
ax5.text(0.05, 0.95, f'Mean: {residuals_svm.mean():.3f}°C\nStd: {residuals_svm.std():.3f}°C', 
         transform=ax5.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 6. Error distribution for best kernel
ax6 = axes[1, 2]
ax6.hist(residuals_svm, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax6.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
ax6.set_xlabel('Residuals (°C)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax6.set_title(f'Error Distribution - {best_svm["kernel"]} kernel', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3, axis='y')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('advanced_results/svm_kernels_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ SVM kernels analysis saved")
plt.close()

# PLOT 2: XGBoost Shuffling Comparison
print("  Generating XGBoost shuffling comparison plots...")
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('XGBoost Performance with Different Data Shuffling', fontsize=18, fontweight='bold')

# 1. R² Comparison
ax1 = axes[0, 0]
x_pos = np.arange(len(xgb_df))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, xgb_df['train_r2'], width, label='Train R²', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x_pos + width/2, xgb_df['test_r2'], width, label='Test R²', alpha=0.8, edgecolor='black')
ax1.set_xlabel('Random State (Shuffle Seed)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
ax1.set_title('R² Score by Shuffle Seed', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(xgb_df['random_state'])
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3, axis='y')
for bar in bars2:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}', 
            ha='center', va='bottom', fontsize=9)

# 2. MAE Comparison
ax2 = axes[0, 1]
bars3 = ax2.bar(x_pos - width/2, xgb_df['train_mae'], width, label='Train MAE', alpha=0.8, edgecolor='black')
bars4 = ax2.bar(x_pos + width/2, xgb_df['test_mae'], width, label='Test MAE', alpha=0.8, edgecolor='black')
ax2.set_xlabel('Random State (Shuffle Seed)', fontsize=12, fontweight='bold')
ax2.set_ylabel('MAE (°C)', fontsize=12, fontweight='bold')
ax2.set_title('Mean Absolute Error by Shuffle Seed', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(xgb_df['random_state'])
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3, axis='y')
for bar in bars4:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', 
            ha='center', va='bottom', fontsize=9)

# 3. Variance in Test R²
ax3 = axes[0, 2]
mean_r2 = xgb_df['test_r2'].mean()
std_r2 = xgb_df['test_r2'].std()
ax3.plot(xgb_df['random_state'], xgb_df['test_r2'], 'o-', linewidth=2, markersize=10, color='darkgreen')
ax3.axhline(y=mean_r2, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_r2:.4f}')
ax3.fill_between(range(len(xgb_df)), mean_r2 - std_r2, mean_r2 + std_r2, alpha=0.2, color='red')
ax3.set_xlabel('Random State', fontsize=12, fontweight='bold')
ax3.set_ylabel('Test R² Score', fontsize=12, fontweight='bold')
ax3.set_title(f'R² Stability (Std: {std_r2:.4f})', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 4. True vs Predicted for best shuffle
ax4 = axes[1, 0]
best_xgb_idx = xgb_df['test_r2'].idxmax()
best_xgb = xgb_results[best_xgb_idx]
y_test_xgb = best_xgb['y_test']
y_pred_xgb = best_xgb['y_pred']
ax4.scatter(y_test_xgb, y_pred_xgb, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
min_val = min(y_test_xgb.min(), y_pred_xgb.min())
max_val = max(y_test_xgb.max(), y_pred_xgb.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('True Temperature (°C)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
ax4.set_title(f'Best Shuffle: seed={best_xgb["random_state"]} (R²={best_xgb["test_r2"]:.4f})', 
              fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)

# 5. Residuals for best shuffle
ax5 = axes[1, 1]
residuals_xgb = y_test_xgb.values - y_pred_xgb
ax5.scatter(y_pred_xgb, residuals_xgb, alpha=0.5, s=20, edgecolors='k', linewidths=0.3)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Temperature (°C)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Residuals (°C)', fontsize=12, fontweight='bold')
ax5.set_title(f'Residual Plot - seed={best_xgb["random_state"]}', fontsize=13, fontweight='bold')
ax5.grid(alpha=0.3)
ax5.text(0.05, 0.95, f'Mean: {residuals_xgb.mean():.3f}°C\nStd: {residuals_xgb.std():.3f}°C', 
         transform=ax5.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# 6. MAE variance across shuffles
ax6 = axes[1, 2]
mean_mae = xgb_df['test_mae'].mean()
std_mae = xgb_df['test_mae'].std()
ax6.plot(xgb_df['random_state'], xgb_df['test_mae'], 's-', linewidth=2, markersize=10, color='darkorange')
ax6.axhline(y=mean_mae, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_mae:.2f}°C')
ax6.fill_between(range(len(xgb_df)), mean_mae - std_mae, mean_mae + std_mae, alpha=0.2, color='blue')
ax6.set_xlabel('Random State', fontsize=12, fontweight='bold')
ax6.set_ylabel('Test MAE (°C)', fontsize=12, fontweight='bold')
ax6.set_title(f'MAE Stability (Std: {std_mae:.2f}°C)', fontsize=13, fontweight='bold')
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('advanced_results/xgboost_shuffling_analysis.png', dpi=300, bbox_inches='tight')
print("  ✓ XGBoost shuffling analysis saved")
plt.close()

# PLOT 3: Combined Comparison
print("  Generating combined comparison plot...")
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('SVM Kernels vs XGBoost Shuffling - Performance Overview', fontsize=16, fontweight='bold')

# 1. Best from each approach
ax1 = axes[0]
best_performers = pd.DataFrame([
    {'Method': f'SVM ({best_svm["kernel"]})', 'R²': best_svm['test_r2'], 'MAE': best_svm['test_mae']},
    {'Method': f'XGBoost (seed={best_xgb["random_state"]})', 'R²': best_xgb['test_r2'], 'MAE': best_xgb['test_mae']}
])
x_pos = np.arange(len(best_performers))
width = 0.35
bars1 = ax1.bar(x_pos - width/2, best_performers['R²'], width, label='R² Score', alpha=0.8, edgecolor='black')
ax1_twin = ax1.twinx()
bars2 = ax1_twin.bar(x_pos + width/2, best_performers['MAE'], width, label='MAE (°C)', 
                     alpha=0.8, edgecolor='black', color='orange')
ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold', color='blue')
ax1_twin.set_ylabel('MAE (°C)', fontsize=12, fontweight='bold', color='orange')
ax1.set_title('Best Performers Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(best_performers['Method'], fontsize=10)
ax1.tick_params(axis='y', labelcolor='blue')
ax1_twin.tick_params(axis='y', labelcolor='orange')
ax1.grid(alpha=0.3, axis='y')
for bar, val in zip(bars1, best_performers['R²']):
    ax1.text(bar.get_x() + bar.get_width()/2., val + 0.01, f'{val:.4f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
for bar, val in zip(bars2, best_performers['MAE']):
    ax1_twin.text(bar.get_x() + bar.get_width()/2., val + 0.1, f'{val:.2f}', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold', color='orange')

# 2. Variability comparison
ax2 = axes[1]
variability_data = pd.DataFrame([
    {'Approach': 'SVM Kernels', 'Metric': 'R²', 'Mean': svm_df['test_r2'].mean(), 'Std': svm_df['test_r2'].std()},
    {'Approach': 'XGBoost Shuffles', 'Metric': 'R²', 'Mean': xgb_df['test_r2'].mean(), 'Std': xgb_df['test_r2'].std()},
    {'Approach': 'SVM Kernels', 'Metric': 'MAE', 'Mean': svm_df['test_mae'].mean(), 'Std': svm_df['test_mae'].std()},
    {'Approach': 'XGBoost Shuffles', 'Metric': 'MAE', 'Mean': xgb_df['test_mae'].mean(), 'Std': xgb_df['test_mae'].std()}
])
labels = ['SVM (R²)', 'XGB (R²)', 'SVM (MAE)', 'XGB (MAE)']
means = variability_data['Mean'].values
stds = variability_data['Std'].values
x_pos = np.arange(len(labels))
bars = ax2.bar(x_pos, stds, alpha=0.7, edgecolor='black', linewidth=1.5)
bars[0].set_color('skyblue')
bars[1].set_color('lightgreen')
bars[2].set_color('lightcoral')
bars[3].set_color('gold')
ax2.set_xlabel('Approach & Metric', fontsize=12, fontweight='bold')
ax2.set_ylabel('Standard Deviation', fontsize=12, fontweight='bold')
ax2.set_title('Variability Across Experiments', fontsize=13, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(labels, fontsize=10)
ax2.grid(alpha=0.3, axis='y')
for bar, std, mean in zip(bars, stds, means):
    ax2.text(bar.get_x() + bar.get_width()/2., std + 0.001, 
            f'Std: {std:.4f}\nMean: {mean:.3f}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig('advanced_results/combined_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Combined comparison saved")
plt.close()

# --- 5. SUMMARY ---
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\nSVM Kernels Summary:")
print(svm_df[['kernel', 'test_r2', 'test_mae']].to_string(index=False))

print("\nXGBoost Shuffling Summary:")
print(xgb_df[['random_state', 'test_r2', 'test_mae']].to_string(index=False))

print(f"\n✓ Best SVM Kernel: {best_svm['kernel']} (R²={best_svm['test_r2']:.4f}, MAE={best_svm['test_mae']:.2f}°C)")
print(f"✓ Best XGBoost Shuffle: seed={best_xgb['random_state']} (R²={best_xgb['test_r2']:.4f}, MAE={best_xgb['test_mae']:.2f}°C)")

print("\n" + "="*80)
print("Generated Files:")
print("  - advanced_results/svm_kernels_comparison.csv")
print("  - advanced_results/xgboost_shuffling_comparison.csv")
print("  - advanced_results/svm_kernels_analysis.png")
print("  - advanced_results/xgboost_shuffling_analysis.png")
print("  - advanced_results/combined_comparison.png")
print("="*80)
