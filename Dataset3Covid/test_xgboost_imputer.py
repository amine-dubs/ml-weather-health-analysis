"""
Test XGBoost Regressor Imputer Performance
Compare IterativeImputer with XGBoostRegressor vs previous methods
Test with and without normalization
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Create output directory
output_dir = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\covid_results_xgboost_imputer'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("XGBoost Regressor Imputation Test")
print("="*80)

# 1. Load original dataset with missing values
print("\n1. Loading original dataset...")
data_path = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\Dataset3.csv'
df_original = pd.read_csv(data_path)

print(f"Dataset shape: {df_original.shape}")
print(f"Missing values: {df_original.isnull().sum().sum()} ({df_original.isnull().sum().sum() / (df_original.shape[0] * df_original.shape[1]) * 100:.2f}%)")

# Separate features and target
X_with_missing = df_original.drop('SARSCov', axis=1)
y = df_original['SARSCov']

# Store column names and original statistics
feature_names = X_with_missing.columns
original_means = X_with_missing.mean()
original_stds = X_with_missing.std()

print(f"Features: {len(feature_names)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# 2. ITERATIVE IMPUTER WITH XGBOOST (NO SCALING)
# ============================================================================

print("\n2. ITERATIVE IMPUTER WITH XGBOOST REGRESSOR (No Scaling)")
print("="*80)

print("\nApplying IterativeImputer with XGBoost estimator...")
xgb_imputer_no_scale = IterativeImputer(
    estimator=xgb.XGBRegressor(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    ),
    max_iter=10,
    random_state=42,
    verbose=0
)

features_xgb_no_scale = pd.DataFrame(
    xgb_imputer_no_scale.fit_transform(features),
    columns=features.columns,
    index=features.index
)

# Calculate changes
xgb_no_scale_changes = {}
for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    new_mean = features_xgb_no_scale[col].mean()
    mean_change = abs((new_mean - orig_mean) / orig_mean) * 100
    xgb_no_scale_changes[col] = mean_change

avg_change_no_scale = sum(xgb_no_scale_changes.values()) / len(xgb_no_scale_changes)
print(f"✓ Imputation complete")
print(f"  Average mean change: {avg_change_no_scale:.2f}%")

# ============================================================================
# 3. ITERATIVE IMPUTER WITH XGBOOST (WITH SCALING)
# ============================================================================

print("\n3. ITERATIVE IMPUTER WITH XGBOOST REGRESSOR (With Scaling)")
print("="*80)

# Scale features first
print("\nScaling features with StandardScaler...")
scaler_before = StandardScaler()
features_scaled = pd.DataFrame(
    scaler_before.fit_transform(features),
    columns=features.columns,
    index=features.index
)

print("Applying IterativeImputer with XGBoost estimator...")
xgb_imputer_scaled = IterativeImputer(
    estimator=xgb.XGBRegressor(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    ),
    max_iter=10,
    random_state=42,
    verbose=0
)

features_xgb_scaled_temp = pd.DataFrame(
    xgb_imputer_scaled.fit_transform(features_scaled),
    columns=features.columns,
    index=features.index
)

# Inverse transform to original scale
features_xgb_scaled = pd.DataFrame(
    scaler_before.inverse_transform(features_xgb_scaled_temp),
    columns=features.columns,
    index=features.index
)

# Calculate changes
xgb_scaled_changes = {}
for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    new_mean = features_xgb_scaled[col].mean()
    mean_change = abs((new_mean - orig_mean) / orig_mean) * 100
    xgb_scaled_changes[col] = mean_change

avg_change_scaled = sum(xgb_scaled_changes.values()) / len(xgb_scaled_changes)
print(f"✓ Imputation complete")
print(f"  Average mean change: {avg_change_scaled:.2f}%")

# ============================================================================
# 4. COMPARISON WITH OTHER METHODS
# ============================================================================

print("\n4. COMPARISON OF ALL IMPUTATION METHODS")
print("="*80)

# Load previous results
from sklearn.impute import KNNImputer

# KNN k=5
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
features_knn = pd.DataFrame(
    knn_imputer.fit_transform(features),
    columns=features.columns
)
knn_changes = {}
for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    new_mean = features_knn[col].mean()
    mean_change = abs((new_mean - orig_mean) / orig_mean) * 100
    knn_changes[col] = mean_change
avg_change_knn = sum(knn_changes.values()) / len(knn_changes)

# MICE (with default BayesianRidge)
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
features_mice = pd.DataFrame(
    mice_imputer.fit_transform(features),
    columns=features.columns
)
mice_changes = {}
for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    new_mean = features_mice[col].mean()
    mean_change = abs((new_mean - orig_mean) / orig_mean) * 100
    mice_changes[col] = mean_change
avg_change_mice = sum(mice_changes.values()) / len(mice_changes)

print(f"\n{'Method':<40} {'Avg Mean Change':>20} {'Status':>15}")
print("-"*80)
print(f"{'KNN (k=5)':<40} {avg_change_knn:>18.2f}% {'':<15}")
print(f"{'MICE (BayesianRidge)':<40} {avg_change_mice:>18.2f}% {'':<15}")
print(f"{'IterativeImputer + XGBoost (No Scale)':<40} {avg_change_no_scale:>18.2f}% {'':<15}")
print(f"{'IterativeImputer + XGBoost (Scaled)':<40} {avg_change_scaled:>18.2f}% {'':<15}")

# Find best
all_methods = {
    'KNN (k=5)': avg_change_knn,
    'MICE (BayesianRidge)': avg_change_mice,
    'XGBoost Imputer (No Scale)': avg_change_no_scale,
    'XGBoost Imputer (Scaled)': avg_change_scaled
}
best_method = min(all_methods.items(), key=lambda x: x[1])

print(f"\n{'='*80}")
print(f"BEST IMPUTATION METHOD: {best_method[0]}")
print(f"Average Mean Change: {best_method[1]:.2f}%")
print(f"{'='*80}")

# ============================================================================
# 5. DETAILED COMPARISON
# ============================================================================

print("\n5. FEATURE-LEVEL COMPARISON")
print("="*80)

comparison_data = []
for col in original_stats.keys():
    comparison_data.append({
        'Feature': col,
        'Missing_%': original_stats[col]['missing_pct'],
        'KNN': knn_changes[col],
        'MICE': mice_changes[col],
        'XGB_NoScale': xgb_no_scale_changes[col],
        'XGB_Scaled': xgb_scaled_changes[col]
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Best_Method'] = comparison_df[['KNN', 'MICE', 'XGB_NoScale', 'XGB_Scaled']].idxmin(axis=1)
comparison_df['Best_Change'] = comparison_df[['KNN', 'MICE', 'XGB_NoScale', 'XGB_Scaled']].min(axis=1)

print("\nTop 10 Features by Best Imputation Quality:")
top_10 = comparison_df.nsmallest(10, 'Best_Change')
print(top_10[['Feature', 'Missing_%', 'Best_Method', 'Best_Change']].to_string(index=False))

print("\nMethod Selection Summary:")
method_counts = comparison_df['Best_Method'].value_counts()
print(method_counts.to_string())

# ============================================================================
# 6. TRAIN MODELS WITH EACH IMPUTATION METHOD
# ============================================================================

print("\n6. TRAINING XGBOOST CLASSIFIER WITH EACH IMPUTATION")
print("="*80)

results = []

# Test each imputation method
imputation_methods = {
    'KNN (k=5)': features_knn,
    'MICE (BayesianRidge)': features_mice,
    'XGBoost Imputer (No Scale)': features_xgb_no_scale,
    'XGBoost Imputer (Scaled)': features_xgb_scaled
}

for method_name, imputed_features in imputation_methods.items():
    print(f"\n{method_name}:")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        imputed_features, target, test_size=0.2, random_state=42, stratify=target
    )
    
    # Train XGBoost classifier (no scaling - tree-based)
    clf = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Imputation_Method': method_name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc
    })
    
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  AUC-ROC:   {auc:.4f} ({auc*100:.2f}%)")

results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)

print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80)
print(results_df.to_string(index=False))

best_result = results_df.iloc[0]
print(f"\n🏆 BEST COMBINATION:")
print(f"   Imputation: {best_result['Imputation_Method']}")
print(f"   AUC-ROC: {best_result['AUC-ROC']:.4f} ({best_result['AUC-ROC']*100:.2f}%)")
print(f"   Accuracy: {best_result['Accuracy']*100:.2f}%")

# ============================================================================
# 7. VISUALIZATIONS
# ============================================================================

print("\n7. GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Imputation Quality Comparison
ax1 = axes[0, 0]
methods = ['KNN\n(k=5)', 'MICE\n(Bayesian)', 'XGBoost\n(No Scale)', 'XGBoost\n(Scaled)']
changes = [avg_change_knn, avg_change_mice, avg_change_no_scale, avg_change_scaled]
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

bars = ax1.bar(methods, changes, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('Average Mean Change (%)', fontweight='bold', fontsize=12)
ax1.set_title('Imputation Quality Comparison\n(Lower is Better)', fontweight='bold', fontsize=14)
ax1.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, changes):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
             f'{val:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 2: Model Performance Comparison
ax2 = axes[0, 1]
x = np.arange(len(results_df))
width = 0.35

acc_vals = results_df['Accuracy'].values
auc_vals = results_df['AUC-ROC'].values

bars1 = ax2.bar(x - width/2, acc_vals, width, label='Accuracy', color='#3498db', alpha=0.8)
bars2 = ax2.bar(x + width/2, auc_vals, width, label='AUC-ROC', color='#2ecc71', alpha=0.8)

ax2.set_ylabel('Score', fontweight='bold', fontsize=12)
ax2.set_title('XGBoost Classifier Performance\nby Imputation Method', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels([m.split('(')[0].strip() for m in results_df['Imputation_Method']], 
                     rotation=45, ha='right', fontsize=10)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.75, 0.95])

# Plot 3: Method Selection by Feature
ax3 = axes[1, 0]
method_counts = comparison_df['Best_Method'].value_counts()
colors_methods = {'KNN': '#3498db', 'MICE': '#e74c3c', 
                  'XGB_NoScale': '#2ecc71', 'XGB_Scaled': '#f39c12'}
colors_plot = [colors_methods.get(m, '#95a5a6') for m in method_counts.index]

bars = ax3.bar(range(len(method_counts)), method_counts.values, color=colors_plot, 
               alpha=0.8, edgecolor='black', linewidth=2)
ax3.set_xticks(range(len(method_counts)))
ax3.set_xticklabels(method_counts.index, rotation=45, ha='right', fontsize=10)
ax3.set_ylabel('Number of Features', fontweight='bold', fontsize=12)
ax3.set_title('Best Imputation Method per Feature\n(32 features total)', 
              fontweight='bold', fontsize=14)
ax3.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, method_counts.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 4: Scaling Impact on XGBoost Imputer
ax4 = axes[1, 1]
features_to_plot = comparison_df.nlargest(15, 'Missing_%')['Feature'].tolist()
x_pos = np.arange(len(features_to_plot))

no_scale_vals = [xgb_no_scale_changes[f] for f in features_to_plot]
scaled_vals = [xgb_scaled_changes[f] for f in features_to_plot]

bars1 = ax4.barh(x_pos - 0.2, no_scale_vals, 0.4, label='No Scaling', 
                 color='#2ecc71', alpha=0.8)
bars2 = ax4.barh(x_pos + 0.2, scaled_vals, 0.4, label='With Scaling', 
                 color='#f39c12', alpha=0.8)

ax4.set_yticks(x_pos)
ax4.set_yticklabels(features_to_plot, fontsize=10)
ax4.set_xlabel('Mean Change (%)', fontweight='bold', fontsize=12)
ax4.set_title('XGBoost Imputer: Scaling Impact\n(Top 15 Missing Features)', 
              fontweight='bold', fontsize=14)
ax4.legend(fontsize=11)
ax4.grid(axis='x', alpha=0.3)
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig('covid_results_hybrid/xgboost_imputer_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: covid_results_hybrid/xgboost_imputer_comparison.png")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\n8. SAVING RESULTS")
print("="*80)

# Save comparison
comparison_df.to_csv('covid_results_hybrid/xgboost_imputer_feature_comparison.csv', index=False)
print("✓ Feature comparison saved: covid_results_hybrid/xgboost_imputer_feature_comparison.csv")

# Save model results
results_df.to_csv('covid_results_hybrid/xgboost_imputer_model_results.csv', index=False)
print("✓ Model results saved: covid_results_hybrid/xgboost_imputer_model_results.csv")

# Save summary
summary = {
    'imputation_quality': {
        'KNN_k5': avg_change_knn,
        'MICE_BayesianRidge': avg_change_mice,
        'XGBoost_Imputer_NoScale': avg_change_no_scale,
        'XGBoost_Imputer_Scaled': avg_change_scaled
    },
    'model_performance': {
        method: {
            'auc': float(results_df[results_df['Imputation_Method']==method]['AUC-ROC'].values[0]),
            'accuracy': float(results_df[results_df['Imputation_Method']==method]['Accuracy'].values[0])
        }
        for method in imputation_methods.keys()
    },
    'best_imputation': best_method[0],
    'best_model': best_result['Imputation_Method']
}

import json
with open('covid_results_hybrid/xgboost_imputer_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Summary saved: covid_results_hybrid/xgboost_imputer_summary.json")

# ============================================================================
# 9. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print(" SUMMARY: XGBoost Regressor for Imputation")
print("="*80)

print(f"\n📊 IMPUTATION QUALITY:")
print(f"  Best: {best_method[0]} - {best_method[1]:.2f}% mean change")
print(f"  XGBoost No Scale: {avg_change_no_scale:.2f}%")
print(f"  XGBoost Scaled: {avg_change_scaled:.2f}%")

scaling_diff = avg_change_scaled - avg_change_no_scale
print(f"\n  Scaling Impact: {scaling_diff:+.2f}% {'(worse)' if scaling_diff > 0 else '(better)'}")

print(f"\n🤖 MODEL PERFORMANCE:")
print(f"  Best: {best_result['Imputation_Method']}")
print(f"  AUC-ROC: {best_result['AUC-ROC']:.4f} ({best_result['AUC-ROC']*100:.2f}%)")
print(f"  Accuracy: {best_result['Accuracy']*100:.2f}%")

print(f"\n💡 KEY INSIGHTS:")
print(f"  1. XGBoost as imputer: {'Better' if avg_change_no_scale < avg_change_mice else 'Worse'} than MICE")
print(f"  2. Scaling for XGBoost imputer: {'Improves' if scaling_diff < 0 else 'Worsens'} quality by {abs(scaling_diff):.2f}%")
print(f"  3. Best features use: {method_counts.to_dict()}")

print("\n" + "="*80)
print("✓ XGBoost Regressor Imputation Analysis Complete!")
print("="*80)
