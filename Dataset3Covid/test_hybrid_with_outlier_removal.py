"""
Hybrid Imputation + Isolation Forest Outlier Removal
Use best hybrid imputation strategy, then apply Isolation Forest outlier detection
Train XGBoost and save if performance improves
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_validate
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
output_dir = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\covid_results_hybrid_outlier_removal'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("HYBRID IMPUTATION + ISOLATION FOREST OUTLIER REMOVAL")
print("Best imputation per feature + Outlier detection")
print("="*80)

# 1. Load original dataset
print("\n1. Loading original dataset...")
data_path = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\Dataset3.csv'
df_original = pd.read_csv(data_path)

print(f"Original dataset shape: {df_original.shape}")
print(f"Missing values: {df_original.isnull().sum().sum()}")

# Separate features and target
X_with_missing = df_original.drop('SARSCov', axis=1)
y = df_original['SARSCov']
feature_names = X_with_missing.columns

# Store original statistics
original_means = X_with_missing.mean()
original_stds = X_with_missing.std()

print(f"Features: {len(feature_names)}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 2. Apply all imputation methods
print("\n2. Applying all imputation methods...")
print("-"*80)

# KNN k=5
print("Applying KNN (k=5)...")
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_knn = pd.DataFrame(
    knn_imputer.fit_transform(X_with_missing),
    columns=feature_names,
    index=X_with_missing.index
)

# MICE with BayesianRidge
print("Applying MICE (BayesianRidge)...")
mice_imputer = IterativeImputer(max_iter=10, random_state=42, verbose=0)
X_mice = pd.DataFrame(
    mice_imputer.fit_transform(X_with_missing),
    columns=feature_names,
    index=X_with_missing.index
)

# XGBoost Imputer (No Scaling)
print("Applying XGBoost Imputer (No Scaling)...")
xgb_imputer_no_scale = IterativeImputer(
    estimator=XGBRegressor(
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
X_xgb_no_scale = pd.DataFrame(
    xgb_imputer_no_scale.fit_transform(X_with_missing),
    columns=feature_names,
    index=X_with_missing.index
)

# XGBoost Imputer (With Scaling)
print("Applying XGBoost Imputer (With Scaling)...")
scaler_temp = StandardScaler()
X_scaled_temp = pd.DataFrame(
    scaler_temp.fit_transform(X_with_missing),
    columns=feature_names,
    index=X_with_missing.index
)

xgb_imputer_scaled = IterativeImputer(
    estimator=XGBRegressor(
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

X_xgb_scaled_temp = pd.DataFrame(
    xgb_imputer_scaled.fit_transform(X_scaled_temp),
    columns=feature_names,
    index=X_with_missing.index
)

X_xgb_scaled = pd.DataFrame(
    scaler_temp.inverse_transform(X_xgb_scaled_temp),
    columns=feature_names,
    index=X_with_missing.index
)

print("✓ All imputations completed!")

# 3. Select best method per feature
print("\n3. Selecting best imputation method per feature...")
print("-"*80)

best_imputation_strategy = {}
feature_selection_details = []

X_best_hybrid = pd.DataFrame(index=X_with_missing.index, columns=feature_names)

for col in feature_names:
    # Calculate mean change for each method
    orig_mean = original_means[col]
    
    if orig_mean == 0:
        orig_mean = 1e-10  # Avoid division by zero
    
    knn_change = abs((X_knn[col].mean() - orig_mean) / orig_mean) * 100
    mice_change = abs((X_mice[col].mean() - orig_mean) / orig_mean) * 100
    xgb_no_scale_change = abs((X_xgb_no_scale[col].mean() - orig_mean) / orig_mean) * 100
    xgb_scaled_change = abs((X_xgb_scaled[col].mean() - orig_mean) / orig_mean) * 100
    
    # Find best method (minimum change)
    changes = {
        'KNN': knn_change,
        'MICE': mice_change,
        'XGB_NoScale': xgb_no_scale_change,
        'XGB_Scaled': xgb_scaled_change
    }
    
    best_method = min(changes.items(), key=lambda x: x[1])
    best_imputation_strategy[col] = best_method[0]
    
    # Select best imputed values for this feature
    if best_method[0] == 'KNN':
        X_best_hybrid[col] = X_knn[col]
    elif best_method[0] == 'MICE':
        X_best_hybrid[col] = X_mice[col]
    elif best_method[0] == 'XGB_NoScale':
        X_best_hybrid[col] = X_xgb_no_scale[col]
    else:  # XGB_Scaled
        X_best_hybrid[col] = X_xgb_scaled[col]
    
    feature_selection_details.append({
        'Feature': col,
        'Original_Mean': orig_mean,
        'KNN_Change_%': knn_change,
        'MICE_Change_%': mice_change,
        'XGB_NoScale_Change_%': xgb_no_scale_change,
        'XGB_Scaled_Change_%': xgb_scaled_change,
        'Best_Method': best_method[0],
        'Best_Change_%': best_method[1]
    })

# Convert to proper numeric type
X_best_hybrid = X_best_hybrid.astype(float)

# Calculate overall quality
avg_mean_change = np.mean([d['Best_Change_%'] for d in feature_selection_details])

print(f"✓ Best hybrid imputation created!")
print(f"  Average mean change: {avg_mean_change:.2f}%")

# Method distribution
method_counts = pd.Series(best_imputation_strategy).value_counts()
print(f"\nMethod distribution:")
for method, count in method_counts.items():
    print(f"  {method}: {count} features ({count/len(feature_names)*100:.1f}%)")

# 4. Baseline model (with hybrid imputation, no outlier removal)
print("\n4. Training baseline model (Hybrid imputation, NO outlier removal)...")
print("-"*80)

xgb_baseline = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

print("Performing 5-fold cross-validation...")
cv_baseline = cross_validate(xgb_baseline, X_best_hybrid, y, cv=cv, scoring=scoring, 
                             return_train_score=False, n_jobs=-1)

baseline_auc = cv_baseline['test_roc_auc'].mean()
baseline_auc_std = cv_baseline['test_roc_auc'].std()
baseline_acc = cv_baseline['test_accuracy'].mean()
baseline_acc_std = cv_baseline['test_accuracy'].std()

print(f"✓ Baseline Results (Hybrid Imputation):")
print(f"  AUC-ROC: {baseline_auc:.4f} (+/- {baseline_auc_std:.4f})")
print(f"  Accuracy: {baseline_acc:.4f} (+/- {baseline_acc_std:.4f})")
print(f"  Samples: {len(X_best_hybrid)}")

# 5. Apply Isolation Forest outlier detection
print("\n5. Applying Isolation Forest outlier detection...")
print("-"*80)

# Scale data for outlier detection
scaler_outlier = StandardScaler()
X_scaled_outlier = scaler_outlier.fit_transform(X_best_hybrid)

# Use best contamination from previous analysis
contamination = 0.15

print(f"Contamination level: {contamination}")
iso_forest = IsolationForest(
    contamination=contamination,
    random_state=42,
    n_jobs=-1
)

outlier_labels = iso_forest.fit_predict(X_scaled_outlier)
inlier_mask = outlier_labels == 1

X_clean = X_best_hybrid[inlier_mask]
y_clean = y[inlier_mask]

samples_removed = len(X_best_hybrid) - len(X_clean)
removal_pct = (samples_removed / len(X_best_hybrid)) * 100

print(f"✓ Outlier detection completed")
print(f"  Original samples: {len(X_best_hybrid)}")
print(f"  Outliers removed: {samples_removed} ({removal_pct:.1f}%)")
print(f"  Clean samples: {len(X_clean)}")
print(f"  Target distribution: {y_clean.value_counts().to_dict()}")

# 6. Train model with outlier removal
print("\n6. Training model with outlier removal...")
print("-"*80)

xgb_clean = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

print("Performing 5-fold cross-validation...")
cv_clean = cross_validate(xgb_clean, X_clean, y_clean, cv=cv, scoring=scoring, 
                          return_train_score=False, n_jobs=-1)

clean_auc = cv_clean['test_roc_auc'].mean()
clean_auc_std = cv_clean['test_roc_auc'].std()
clean_acc = cv_clean['test_accuracy'].mean()
clean_acc_std = cv_clean['test_accuracy'].std()
clean_precision = cv_clean['test_precision'].mean()
clean_recall = cv_clean['test_recall'].mean()
clean_f1 = cv_clean['test_f1'].mean()

print(f"✓ Results with Outlier Removal:")
print(f"  AUC-ROC: {clean_auc:.4f} (+/- {clean_auc_std:.4f})")
print(f"  Accuracy: {clean_acc:.4f} (+/- {clean_acc_std:.4f})")
print(f"  Precision: {clean_precision:.4f}")
print(f"  Recall: {clean_recall:.4f}")
print(f"  F1-Score: {clean_f1:.4f}")

# Train on full clean data for feature importance
xgb_clean.fit(X_clean, y_clean)
y_pred = xgb_clean.predict(X_clean)
y_pred_proba = xgb_clean.predict_proba(X_clean)[:, 1]

full_auc = roc_auc_score(y_clean, y_pred_proba)
full_acc = accuracy_score(y_clean, y_pred)

# 7. Compare results
print("\n7. Comparison with Previous Best Methods...")
print("="*80)

# Previous best results
previous_results = {
    'KNN k=5 (Original)': {'AUC': 0.8975, 'Accuracy': 0.8391, 'Imputation': 2.20, 'Outlier_Removal': False},
    'KNN + IsolationForest': {'AUC': 0.8956, 'Accuracy': 0.8217, 'Imputation': 2.20, 'Outlier_Removal': True},
    'Hybrid (All Methods)': {'AUC': 0.8842, 'Accuracy': 0.8226, 'Imputation': 0.63, 'Outlier_Removal': False}
}

print(f"\n{'Method':<35} {'AUC-ROC':<12} {'Accuracy':<12} {'Imputation':<15} {'Outliers':<10}")
print("-"*80)
for method, metrics in previous_results.items():
    imp_str = f"{metrics['Imputation']:.2f}%"
    outlier_str = "Yes" if metrics['Outlier_Removal'] else "No"
    print(f"{method:<35} {metrics['AUC']:<12.4f} {metrics['Accuracy']:<12.4f} {imp_str:<15} {outlier_str:<10}")

print(f"{'Hybrid + IsolationForest (NEW)':<35} {clean_auc:<12.4f} {clean_acc:<12.4f} {avg_mean_change:<15.2f}% {'Yes':<10}")
print(f"{'Hybrid Baseline (no outliers)':<35} {baseline_auc:<12.4f} {baseline_acc:<12.4f} {avg_mean_change:<15.2f}% {'No':<10}")
print("-"*80)

# Calculate improvements
best_prev_auc = max([r['AUC'] for r in previous_results.values()])
best_prev_acc = max([r['Accuracy'] for r in previous_results.values()])

auc_improvement = (clean_auc - best_prev_auc) * 100
acc_improvement = (clean_acc - best_prev_acc) * 100
auc_vs_baseline = (clean_auc - baseline_auc) * 100
acc_vs_baseline = (clean_acc - baseline_acc) * 100

print(f"\n📊 IMPROVEMENTS:")
print(f"  vs Best Overall (KNN k=5): AUC {auc_improvement:+.2f} pp, Acc {acc_improvement:+.2f} pp")
print(f"  vs Hybrid Baseline: AUC {auc_vs_baseline:+.2f} pp, Acc {acc_vs_baseline:+.2f} pp")

# 8. Visualizations
print("\n8. Generating visualizations...")
print("-"*80)

fig = plt.figure(figsize=(20, 14))

# Plot 1: ROC Curve
ax1 = plt.subplot(2, 3, 1)
fpr, tpr, _ = roc_curve(y_clean, y_pred_proba)
plt.plot(fpr, tpr, label=f'Hybrid+IsolationForest (AUC={full_auc:.4f})', 
         linewidth=2.5, color='#e74c3c')
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
plt.title('ROC Curve - Hybrid Imputation + Outlier Removal', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

# Plot 2: Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y_clean, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False, 
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title(f'Confusion Matrix\nAccuracy: {full_acc:.4f}', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=11, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')

# Plot 3: AUC Comparison
ax3 = plt.subplot(2, 3, 3)
methods = ['KNN k=5', 'KNN +\nIsolation', 'Hybrid\nBaseline', 'Hybrid +\nIsolation']
aucs = [0.8975, 0.8956, baseline_auc, clean_auc]
colors = ['#3498db', '#2ecc71', '#95a5a6', '#e74c3c']

bars = plt.bar(methods, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('AUC-ROC', fontsize=11, fontweight='bold')
plt.title('AUC-ROC Comparison', fontsize=13, fontweight='bold')
plt.ylim([0.88, 0.91])
plt.axhline(y=best_prev_auc, color='red', linestyle='--', alpha=0.5, linewidth=2)

for bar, auc in zip(bars, aucs):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Plot 4: Accuracy Comparison
ax4 = plt.subplot(2, 3, 4)
accs = [0.8391, 0.8217, baseline_acc, clean_acc]

bars = plt.bar(methods, accs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('Accuracy', fontsize=11, fontweight='bold')
plt.title('Accuracy Comparison', fontsize=13, fontweight='bold')
plt.ylim([0.81, 0.85])
plt.axhline(y=best_prev_acc, color='red', linestyle='--', alpha=0.5, linewidth=2)

for bar, acc in zip(bars, accs):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Plot 5: Data Removed
ax5 = plt.subplot(2, 3, 5)
removal_data = {
    'KNN k=5': 0,
    'KNN +\nIsolation': 15.0,
    'Hybrid\nBaseline': 0,
    'Hybrid +\nIsolation': removal_pct
}

bars = plt.bar(removal_data.keys(), removal_data.values(), color=colors, 
               alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('Samples Removed (%)', fontsize=11, fontweight='bold')
plt.title('Outlier Removal Rate', fontsize=13, fontweight='bold')

for bar, val in zip(bars, removal_data.values()):
    if val > 0:
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.grid(axis='y', alpha=0.3)

# Plot 6: Feature Importance (Top 15)
ax6 = plt.subplot(2, 3, 6)
feature_importance = xgb_clean.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance,
    'Imputation_Method': [best_imputation_strategy[f] for f in feature_names]
}).sort_values('Importance', ascending=False).head(15)

colors_imp = [{'KNN': '#3498db', 'MICE': '#e74c3c', 'XGB_NoScale': '#2ecc71', 
               'XGB_Scaled': '#f39c12'}.get(m, '#95a5a6') 
              for m in importance_df['Imputation_Method']]

plt.barh(range(len(importance_df)), importance_df['Importance'], color=colors_imp, alpha=0.8)
plt.yticks(range(len(importance_df)), importance_df['Feature'], fontsize=10)
plt.xlabel('Importance', fontsize=11, fontweight='bold')
plt.title('Top 15 Feature Importance\n(color = imputation method)', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'hybrid_outlier_removal_results.png'), dpi=300, bbox_inches='tight')
print(f"✓ Visualizations saved: {output_dir}/hybrid_outlier_removal_results.png")
plt.close()

# 9. Save results
print("\n9. Saving results...")
print("-"*80)

# Save comparison results
results_df = pd.DataFrame([
    {
        'Method': 'KNN k=5 (Original)',
        'AUC': 0.8975,
        'Accuracy': 0.8391,
        'Imputation_Quality_%': 2.20,
        'Outlier_Removal': 'No',
        'Samples': 1736,
        'Removed': 0
    },
    {
        'Method': 'KNN + IsolationForest',
        'AUC': 0.8956,
        'Accuracy': 0.8217,
        'Imputation_Quality_%': 2.20,
        'Outlier_Removal': 'Yes (15%)',
        'Samples': 1475,
        'Removed': 261
    },
    {
        'Method': 'Hybrid Baseline',
        'AUC': baseline_auc,
        'Accuracy': baseline_acc,
        'Imputation_Quality_%': avg_mean_change,
        'Outlier_Removal': 'No',
        'Samples': len(X_best_hybrid),
        'Removed': 0
    },
    {
        'Method': 'Hybrid + IsolationForest',
        'AUC': clean_auc,
        'Accuracy': clean_acc,
        'Imputation_Quality_%': avg_mean_change,
        'Outlier_Removal': f'Yes ({removal_pct:.1f}%)',
        'Samples': len(X_clean),
        'Removed': samples_removed
    }
])

results_df.to_csv(os.path.join(output_dir, 'comprehensive_comparison.csv'), index=False)
print(f"✓ Comparison saved: {output_dir}/comprehensive_comparison.csv")

# Save feature selection details
selection_df = pd.DataFrame(feature_selection_details)
selection_df.to_csv(os.path.join(output_dir, 'feature_selection_details.csv'), index=False)

# 10. Decision: Save if improved
print("\n10. Evaluating save criteria...")
print("="*80)

save_model = False
save_reason = []

if clean_auc > best_prev_auc:
    save_model = True
    save_reason.append(f"AUC improved vs best overall: {clean_auc:.4f} > {best_prev_auc:.4f} ({auc_improvement:+.2f} pp)")

if clean_acc > best_prev_acc:
    save_model = True
    save_reason.append(f"Accuracy improved vs best overall: {clean_acc:.4f} > {best_prev_acc:.4f} ({acc_improvement:+.2f} pp)")

if clean_auc > baseline_auc:
    save_model = True
    save_reason.append(f"AUC improved vs Hybrid baseline: {clean_auc:.4f} > {baseline_auc:.4f} ({auc_vs_baseline:+.2f} pp)")

if save_model:
    print("✓ SAVE CRITERIA MET!")
    print("\nReasons:")
    for reason in save_reason:
        print(f"  • {reason}")
    
    print(f"\nSaving models and artifacts to: {output_dir}")
    
    # Save model
    joblib.dump(xgb_clean, os.path.join(output_dir, 'best_hybrid_outlier_model.joblib'))
    print("  ✓ Model saved: best_hybrid_outlier_model.joblib")
    
    # Save imputers
    joblib.dump(knn_imputer, os.path.join(output_dir, 'knn_imputer.joblib'))
    joblib.dump(mice_imputer, os.path.join(output_dir, 'mice_imputer.joblib'))
    joblib.dump(xgb_imputer_no_scale, os.path.join(output_dir, 'xgb_imputer_no_scale.joblib'))
    joblib.dump(xgb_imputer_scaled, os.path.join(output_dir, 'xgb_imputer_scaled.joblib'))
    joblib.dump(scaler_temp, os.path.join(output_dir, 'scaler_for_xgb_imputer.joblib'))
    print("  ✓ All imputers saved")
    
    # Save outlier detector and scaler
    joblib.dump(iso_forest, os.path.join(output_dir, 'isolation_forest.joblib'))
    joblib.dump(scaler_outlier, os.path.join(output_dir, 'scaler_outlier.joblib'))
    print("  ✓ Outlier detector saved")
    
    # Save clean dataset
    X_clean_with_target = X_clean.copy()
    X_clean_with_target['SARSCov'] = y_clean
    X_clean_with_target.to_csv(os.path.join(output_dir, 'dataset_hybrid_clean.csv'), index=False)
    print("  ✓ Clean dataset saved")
    
    # Save configuration
    config = {
        'imputation_strategy': best_imputation_strategy,
        'avg_imputation_change_%': avg_mean_change,
        'outlier_method': 'IsolationForest',
        'contamination': contamination,
        'original_samples': len(X_best_hybrid),
        'samples_removed': samples_removed,
        'removal_percentage': removal_pct,
        'final_samples': len(X_clean),
        'cv_auc': clean_auc,
        'cv_auc_std': clean_auc_std,
        'cv_accuracy': clean_acc,
        'cv_accuracy_std': clean_acc_std,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    import json
    with open(os.path.join(output_dir, 'pipeline_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save production guide
    with open(os.path.join(output_dir, 'PRODUCTION_PIPELINE.txt'), 'w') as f:
        f.write("HYBRID IMPUTATION + ISOLATION FOREST - PRODUCTION PIPELINE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Cross-Validated AUC: {clean_auc:.4f} (+/- {clean_auc_std:.4f})\n")
        f.write(f"  Cross-Validated Accuracy: {clean_acc:.4f} (+/- {clean_acc_std:.4f})\n")
        f.write(f"  Imputation Quality: {avg_mean_change:.2f}% avg mean change\n")
        f.write(f"  Outlier Removal: {removal_pct:.1f}% of data\n\n")
        f.write(f"Improvements:\n")
        for reason in save_reason:
            f.write(f"  • {reason}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("DEPLOYMENT STEPS:\n\n")
        f.write("1. Load all imputers and apply hybrid strategy:\n")
        f.write("   - KNN, MICE, XGBoost (scaled/unscaled)\n")
        f.write("   - Select best imputed value per feature\n\n")
        f.write("2. Apply Isolation Forest outlier detection:\n")
        f.write("   - Scale with scaler_outlier.joblib\n")
        f.write(f"   - Detect with contamination={contamination}\n")
        f.write("   - Remove outliers\n\n")
        f.write("3. Predict with best_hybrid_outlier_model.joblib\n\n")
        f.write("="*80 + "\n\n")
        f.write("FEATURE IMPUTATION STRATEGY:\n\n")
        for col, method in sorted(best_imputation_strategy.items()):
            change = selection_df[selection_df['Feature']==col]['Best_Change_%'].values[0]
            f.write(f"  {col:<20} -> {method:<15} (change: {change:.2f}%)\n")
    
    print("  ✓ Production guide saved: PRODUCTION_PIPELINE.txt")
    
    print("\n" + "="*80)
    print("✓ ALL MODELS AND ARTIFACTS SAVED SUCCESSFULLY!")
    print("="*80)
    
else:
    print("✗ SAVE CRITERIA NOT MET")
    print(f"\nCurrent results:")
    print(f"  AUC: {clean_auc:.4f} (Best overall: {best_prev_auc:.4f})")
    print(f"  Accuracy: {clean_acc:.4f} (Best overall: {best_prev_acc:.4f})")
    print("\nResults saved for analysis but models not marked as 'best'.")

# 11. Final Summary
print("\n" + "="*80)
print("FINAL SUMMARY - HYBRID IMPUTATION + OUTLIER REMOVAL")
print("="*80)

print(f"\n📊 PIPELINE CONFIGURATION:")
print(f"  Imputation: Best of 4 methods per feature")
print(f"    • KNN: {method_counts.get('KNN', 0)} features")
print(f"    • MICE: {method_counts.get('MICE', 0)} features")
print(f"    • XGBoost (no scale): {method_counts.get('XGB_NoScale', 0)} features")
print(f"    • XGBoost (scaled): {method_counts.get('XGB_Scaled', 0)} features")
print(f"  Imputation quality: {avg_mean_change:.2f}% avg mean change")
print(f"  Outlier detection: Isolation Forest (contamination={contamination})")
print(f"  Samples removed: {samples_removed} ({removal_pct:.1f}%)")

print(f"\n🤖 MODEL PERFORMANCE:")
print(f"  Cross-Validated AUC: {clean_auc:.4f} (+/- {clean_auc_std:.4f})")
print(f"  Cross-Validated Accuracy: {clean_acc:.4f} (+/- {clean_acc_std:.4f})")
print(f"  Precision: {clean_precision:.4f}")
print(f"  Recall: {clean_recall:.4f}")
print(f"  F1-Score: {clean_f1:.4f}")

print(f"\n📈 COMPARISON:")
print(f"  vs Best Overall (KNN k=5):")
print(f"    AUC: {auc_improvement:+.2f} pp")
print(f"    Accuracy: {acc_improvement:+.2f} pp")
print(f"  vs Hybrid Baseline:")
print(f"    AUC: {auc_vs_baseline:+.2f} pp")
print(f"    Accuracy: {acc_vs_baseline:+.2f} pp")

if save_model:
    print(f"\n✅ RECOMMENDATION: Use Hybrid + IsolationForest pipeline")
    print(f"   Location: {output_dir}")
else:
    print(f"\n⚠️  Best method remains: KNN k=5 (AUC: 0.8975)")

print("\n" + "="*80)
print("✓ HYBRID IMPUTATION + OUTLIER REMOVAL ANALYSIS COMPLETE!")
print("="*80)
