"""
Hybrid Best-of-All Imputation Strategy
Select best imputation method per feature from KNN, MICE, XGBoost (scaled & unscaled)
Train XGBoost classifier and save if AUC or Accuracy improves
"""

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from xgboost import XGBRegressor, XGBClassifier
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
output_dir = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\covid_results_best_hybrid'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("BEST HYBRID IMPUTATION STRATEGY")
print("Selecting best method per feature from: KNN, MICE, XGBoost (scaled/unscaled)")
print("="*80)

# 1. Load original dataset
print("\n1. Loading original dataset...")
data_path = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\Dataset3.csv'
df_original = pd.read_csv(data_path)

print(f"Dataset shape: {df_original.shape}")
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

# Save feature selection details
selection_df = pd.DataFrame(feature_selection_details)
selection_df.to_csv(os.path.join(output_dir, 'feature_selection_details.csv'), index=False)

# Save imputation strategy
with open(os.path.join(output_dir, 'imputation_strategy.txt'), 'w') as f:
    f.write("Best Hybrid Imputation Strategy\n")
    f.write("="*80 + "\n\n")
    f.write(f"Average Mean Change: {avg_mean_change:.2f}%\n\n")
    f.write("Method Distribution:\n")
    for method, count in method_counts.items():
        f.write(f"  {method}: {count} features ({count/len(feature_names)*100:.1f}%)\n")
    f.write("\n" + "="*80 + "\n\n")
    f.write("Feature-Level Strategy:\n")
    for col, method in sorted(best_imputation_strategy.items()):
        change = selection_df[selection_df['Feature']==col]['Best_Change_%'].values[0]
        f.write(f"  {col:<20} -> {method:<15} (change: {change:.2f}%)\n")

print(f"\n✓ Strategy saved to: {output_dir}/imputation_strategy.txt")

# 4. Train XGBoost with 5-fold cross-validation
print("\n4. Training XGBoost Classifier with Best Hybrid Imputation...")
print("-"*80)

# XGBoost classifier
xgb_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

# 5-fold stratified cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1',
    'roc_auc': 'roc_auc'
}

print("Performing 5-fold cross-validation...")
cv_results = cross_validate(xgb_model, X_best_hybrid, y, cv=cv, scoring=scoring, 
                            return_train_score=False, n_jobs=-1)

# Calculate mean and std for each metric
cv_accuracy = cv_results['test_accuracy'].mean()
cv_accuracy_std = cv_results['test_accuracy'].std()
cv_precision = cv_results['test_precision'].mean()
cv_recall = cv_results['test_recall'].mean()
cv_f1 = cv_results['test_f1'].mean()
cv_auc = cv_results['test_roc_auc'].mean()
cv_auc_std = cv_results['test_roc_auc'].std()

print(f"\nCross-Validation Results (5-fold):")
print(f"  Accuracy:  {cv_accuracy:.4f} (+/- {cv_accuracy_std:.4f})")
print(f"  Precision: {cv_precision:.4f}")
print(f"  Recall:    {cv_recall:.4f}")
print(f"  F1-Score:  {cv_f1:.4f}")
print(f"  AUC-ROC:   {cv_auc:.4f} (+/- {cv_auc_std:.4f})")

# Train on full dataset for detailed metrics
print("\nTraining on full dataset...")
xgb_model.fit(X_best_hybrid, y)
y_pred = xgb_model.predict(X_best_hybrid)
y_pred_proba = xgb_model.predict_proba(X_best_hybrid)[:, 1]

# Full dataset metrics
full_accuracy = accuracy_score(y, y_pred)
full_precision = precision_score(y, y_pred)
full_recall = recall_score(y, y_pred)
full_f1 = f1_score(y, y_pred)
full_auc = roc_auc_score(y, y_pred_proba)

print(f"\nFull Dataset Results:")
print(f"  Accuracy:  {full_accuracy:.4f} ({full_accuracy*100:.2f}%)")
print(f"  Precision: {full_precision:.4f}")
print(f"  Recall:    {full_recall:.4f}")
print(f"  F1-Score:  {full_f1:.4f}")
print(f"  AUC-ROC:   {full_auc:.4f} ({full_auc*100:.2f}%)")

# 5. Compare with previous best results
print("\n5. Comparison with Previous Best Methods...")
print("-"*80)

# Previous best results
previous_results = {
    'KNN k=5 (Original)': {'AUC': 0.8975, 'Accuracy': 0.8391, 'Imputation': 2.20},
    'Hybrid (KNN+MICE)': {'AUC': 0.8931, 'Accuracy': 0.8218, 'Imputation': 0.72},
    'XGBoost No Scaling': {'AUC': 0.8854, 'Accuracy': 0.8333, 'Imputation': None}
}

print(f"\n{'Method':<30} {'AUC-ROC':<12} {'Accuracy':<12} {'Imputation Quality':<20}")
print("-"*80)
for method, metrics in previous_results.items():
    imp_str = f"{metrics['Imputation']:.2f}%" if metrics['Imputation'] else "N/A"
    print(f"{method:<30} {metrics['AUC']:<12.4f} {metrics['Accuracy']:<12.4f} {imp_str:<20}")

print(f"{'Best Hybrid (NEW)':<30} {cv_auc:<12.4f} {cv_accuracy:<12.4f} {avg_mean_change:<20.2f}%")
print("-"*80)

# Calculate improvements
best_prev_auc = max([r['AUC'] for r in previous_results.values()])
best_prev_acc = max([r['Accuracy'] for r in previous_results.values()])
best_prev_imp = min([r['Imputation'] for r in previous_results.values() if r['Imputation']])

auc_improvement = (cv_auc - best_prev_auc) * 100
acc_improvement = (cv_accuracy - best_prev_acc) * 100
imp_improvement = ((best_prev_imp - avg_mean_change) / best_prev_imp) * 100

print(f"\nImprovement vs Best Previous:")
print(f"  AUC-ROC: {auc_improvement:+.2f} pp {'✓ IMPROVED' if auc_improvement > 0 else '✗ Not improved'}")
print(f"  Accuracy: {acc_improvement:+.2f} pp {'✓ IMPROVED' if acc_improvement > 0 else '✗ Not improved'}")
print(f"  Imputation: {imp_improvement:+.2f}% {'✓ IMPROVED' if imp_improvement > 0 else '✗ Not improved'}")

# 6. Generate visualizations
print("\n6. Generating visualizations...")
print("-"*80)

fig = plt.figure(figsize=(20, 12))

# Plot 1: ROC Curve
ax1 = plt.subplot(2, 3, 1)
fpr, tpr, _ = roc_curve(y, y_pred_proba)
plt.plot(fpr, tpr, label=f'Best Hybrid (AUC = {full_auc:.4f})', linewidth=2.5, color='#e74c3c')
plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
plt.title('ROC Curve - Best Hybrid Imputation', fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)

# Plot 2: Confusion Matrix
ax2 = plt.subplot(2, 3, 2)
cm = confusion_matrix(y, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='RdYlGn', cbar=False, 
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title(f'Confusion Matrix\nAccuracy: {full_accuracy:.4f}', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=11, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')

# Plot 3: Method Distribution
ax3 = plt.subplot(2, 3, 3)
colors_methods = {'KNN': '#3498db', 'MICE': '#e74c3c', 
                  'XGB_NoScale': '#2ecc71', 'XGB_Scaled': '#f39c12'}
colors_plot = [colors_methods.get(m, '#95a5a6') for m in method_counts.index]

bars = plt.bar(range(len(method_counts)), method_counts.values, color=colors_plot, 
               alpha=0.8, edgecolor='black', linewidth=2)
plt.xticks(range(len(method_counts)), method_counts.index, rotation=45, ha='right', fontsize=10)
plt.ylabel('Number of Features', fontsize=11, fontweight='bold')
plt.title('Imputation Method Distribution\n(32 features total)', fontsize=13, fontweight='bold')
plt.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, method_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
             str(val), ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 4: AUC Comparison
ax4 = plt.subplot(2, 3, 4)
all_methods = ['KNN k=5', 'Hybrid\n(KNN+MICE)', 'XGBoost\nNo Scale', 'Best Hybrid\n(NEW)']
all_aucs = [0.8975, 0.8931, 0.8854, cv_auc]
colors_auc = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

bars = plt.bar(all_methods, all_aucs, color=colors_auc, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('AUC-ROC', fontsize=11, fontweight='bold')
plt.title('AUC-ROC Comparison Across Methods', fontsize=13, fontweight='bold')
plt.ylim([0.88, 0.91])
plt.axhline(y=best_prev_auc, color='red', linestyle='--', alpha=0.5, linewidth=2, 
            label=f'Previous Best ({best_prev_auc:.4f})')

for bar, auc in zip(bars, all_aucs):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{auc:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)

# Plot 5: Imputation Quality Comparison
ax5 = plt.subplot(2, 3, 5)
imp_methods = ['KNN k=5', 'Hybrid\n(KNN+MICE)', 'Best Hybrid\n(NEW)']
imp_qualities = [2.20, 0.72, avg_mean_change]
colors_imp = ['#3498db', '#2ecc71', '#e74c3c']

bars = plt.bar(imp_methods, imp_qualities, color=colors_imp, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('Avg Mean Change (%)', fontsize=11, fontweight='bold')
plt.title('Imputation Quality Comparison\n(Lower is Better)', fontsize=13, fontweight='bold')

for bar, quality in zip(bars, imp_qualities):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
             f'{quality:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.xticks(rotation=0)

# Plot 6: Feature Importance (Top 15)
ax6 = plt.subplot(2, 3, 6)
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False).head(15)

plt.barh(range(len(importance_df)), importance_df['Importance'], color='steelblue', alpha=0.8)
plt.yticks(range(len(importance_df)), importance_df['Feature'], fontsize=10)
plt.xlabel('Importance', fontsize=11, fontweight='bold')
plt.title('Top 15 Feature Importance', fontsize=13, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'best_hybrid_results.png'), dpi=300, bbox_inches='tight')
print(f"✓ Visualizations saved: {output_dir}/best_hybrid_results.png")
plt.close()

# 7. Save comprehensive results
print("\n7. Saving comprehensive results...")
print("-"*80)

# Save all results
results_summary = pd.DataFrame([{
    'Method': 'Best Hybrid (All Methods)',
    'CV_AUC': cv_auc,
    'CV_AUC_Std': cv_auc_std,
    'CV_Accuracy': cv_accuracy,
    'CV_Accuracy_Std': cv_accuracy_std,
    'Full_AUC': full_auc,
    'Full_Accuracy': full_accuracy,
    'Precision': full_precision,
    'Recall': full_recall,
    'F1': full_f1,
    'Imputation_Quality_%': avg_mean_change,
    'KNN_Features': method_counts.get('KNN', 0),
    'MICE_Features': method_counts.get('MICE', 0),
    'XGB_NoScale_Features': method_counts.get('XGB_NoScale', 0),
    'XGB_Scaled_Features': method_counts.get('XGB_Scaled', 0)
}])

results_summary.to_csv(os.path.join(output_dir, 'best_hybrid_summary.csv'), index=False)
print(f"✓ Results summary saved: {output_dir}/best_hybrid_summary.csv")

# Save feature importance
importance_df_full = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance,
    'Imputation_Method': [best_imputation_strategy[f] for f in feature_names]
}).sort_values('Importance', ascending=False)

importance_df_full.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
print(f"✓ Feature importance saved: {output_dir}/feature_importance.csv")

# 8. Decision: Save models if improved
print("\n8. Evaluating save criteria...")
print("="*80)

save_model = False
save_reason = []

# Check improvements
if cv_auc > best_prev_auc:
    save_model = True
    save_reason.append(f"AUC improved: {cv_auc:.4f} > {best_prev_auc:.4f} ({auc_improvement:+.2f} pp)")

if cv_accuracy > best_prev_acc:
    save_model = True
    save_reason.append(f"Accuracy improved: {cv_accuracy:.4f} > {best_prev_acc:.4f} ({acc_improvement:+.2f} pp)")

if avg_mean_change < best_prev_imp:
    save_reason.append(f"Imputation quality improved: {avg_mean_change:.2f}% < {best_prev_imp:.2f}% ({imp_improvement:+.2f}%)")
    if cv_auc >= best_prev_auc * 0.995:  # Within 0.5% of best
        save_model = True

if save_model:
    print("✓ SAVE CRITERIA MET!")
    print("\nReasons:")
    for reason in save_reason:
        print(f"  • {reason}")
    
    print(f"\nSaving models and artifacts to: {output_dir}")
    
    # Save XGBoost model
    joblib.dump(xgb_model, os.path.join(output_dir, 'best_hybrid_xgboost_model.joblib'))
    print("  ✓ Model saved: best_hybrid_xgboost_model.joblib")
    
    # Save imputed dataset
    X_best_hybrid_with_target = X_best_hybrid.copy()
    X_best_hybrid_with_target['SARSCov'] = y
    X_best_hybrid_with_target.to_csv(os.path.join(output_dir, 'dataset_best_hybrid_imputed.csv'), index=False)
    print("  ✓ Imputed dataset saved: dataset_best_hybrid_imputed.csv")
    
    # Save all imputers for production use
    joblib.dump(knn_imputer, os.path.join(output_dir, 'knn_imputer.joblib'))
    joblib.dump(mice_imputer, os.path.join(output_dir, 'mice_imputer.joblib'))
    joblib.dump(xgb_imputer_no_scale, os.path.join(output_dir, 'xgb_imputer_no_scale.joblib'))
    joblib.dump(xgb_imputer_scaled, os.path.join(output_dir, 'xgb_imputer_scaled.joblib'))
    joblib.dump(scaler_temp, os.path.join(output_dir, 'scaler_for_xgb_scaled_imputer.joblib'))
    print("  ✓ All imputers saved")
    
    # Save production pipeline config
    with open(os.path.join(output_dir, 'PRODUCTION_PIPELINE.txt'), 'w') as f:
        f.write("BEST HYBRID IMPUTATION - PRODUCTION PIPELINE\n")
        f.write("="*80 + "\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Cross-Validated AUC: {cv_auc:.4f} (+/- {cv_auc_std:.4f})\n")
        f.write(f"  Cross-Validated Accuracy: {cv_accuracy:.4f} (+/- {cv_accuracy_std:.4f})\n")
        f.write(f"  Imputation Quality: {avg_mean_change:.2f}% avg mean change\n\n")
        f.write(f"Improvements:\n")
        for reason in save_reason:
            f.write(f"  • {reason}\n")
        f.write("\n" + "="*80 + "\n\n")
        f.write("USAGE INSTRUCTIONS:\n\n")
        f.write("1. Load all imputers:\n")
        f.write("   knn_imputer = joblib.load('knn_imputer.joblib')\n")
        f.write("   mice_imputer = joblib.load('mice_imputer.joblib')\n")
        f.write("   xgb_imputer_no_scale = joblib.load('xgb_imputer_no_scale.joblib')\n")
        f.write("   xgb_imputer_scaled = joblib.load('xgb_imputer_scaled.joblib')\n")
        f.write("   scaler = joblib.load('scaler_for_xgb_scaled_imputer.joblib')\n\n")
        f.write("2. For each feature, apply the selected imputation method:\n")
        f.write("   (See imputation_strategy.txt for feature-method mapping)\n\n")
        f.write("3. Load model:\n")
        f.write("   model = joblib.load('best_hybrid_xgboost_model.joblib')\n\n")
        f.write("4. Predict:\n")
        f.write("   predictions = model.predict(X_imputed)\n")
        f.write("   probabilities = model.predict_proba(X_imputed)\n\n")
        f.write("="*80 + "\n\n")
        f.write("FEATURE-LEVEL IMPUTATION STRATEGY:\n\n")
        for col, method in sorted(best_imputation_strategy.items()):
            change = selection_df[selection_df['Feature']==col]['Best_Change_%'].values[0]
            f.write(f"  {col:<20} -> {method:<15} (change: {change:.2f}%)\n")
    
    print("  ✓ Production pipeline guide saved: PRODUCTION_PIPELINE.txt")
    
    # Create deployment script
    deployment_script = """
# Deployment Script - Best Hybrid Imputation Model
import pandas as pd
import joblib

def load_best_hybrid_pipeline(model_dir):
    '''Load all components of the best hybrid imputation pipeline'''
    knn_imputer = joblib.load(f'{model_dir}/knn_imputer.joblib')
    mice_imputer = joblib.load(f'{model_dir}/mice_imputer.joblib')
    xgb_imputer_no_scale = joblib.load(f'{model_dir}/xgb_imputer_no_scale.joblib')
    xgb_imputer_scaled = joblib.load(f'{model_dir}/xgb_imputer_scaled.joblib')
    scaler = joblib.load(f'{model_dir}/scaler_for_xgb_scaled_imputer.joblib')
    model = joblib.load(f'{model_dir}/best_hybrid_xgboost_model.joblib')
    
    # Load imputation strategy
    strategy = {}
    with open(f'{model_dir}/imputation_strategy.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if '->' in line:
                parts = line.strip().split('->')
                feature = parts[0].strip()
                method = parts[1].split('(')[0].strip()
                strategy[feature] = method
    
    return {
        'knn_imputer': knn_imputer,
        'mice_imputer': mice_imputer,
        'xgb_imputer_no_scale': xgb_imputer_no_scale,
        'xgb_imputer_scaled': xgb_imputer_scaled,
        'scaler': scaler,
        'model': model,
        'strategy': strategy
    }

def predict_with_best_hybrid(X_new, pipeline):
    '''Apply best hybrid imputation and predict'''
    # Apply each imputation method
    X_knn = pd.DataFrame(pipeline['knn_imputer'].transform(X_new), columns=X_new.columns)
    X_mice = pd.DataFrame(pipeline['mice_imputer'].transform(X_new), columns=X_new.columns)
    X_xgb_no_scale = pd.DataFrame(pipeline['xgb_imputer_no_scale'].transform(X_new), columns=X_new.columns)
    
    X_scaled = pipeline['scaler'].transform(X_new)
    X_xgb_scaled_temp = pipeline['xgb_imputer_scaled'].transform(X_scaled)
    X_xgb_scaled = pd.DataFrame(pipeline['scaler'].inverse_transform(X_xgb_scaled_temp), columns=X_new.columns)
    
    # Combine using strategy
    X_imputed = pd.DataFrame(index=X_new.index, columns=X_new.columns)
    for col in X_new.columns:
        method = pipeline['strategy'].get(col, 'KNN')
        if method == 'KNN':
            X_imputed[col] = X_knn[col]
        elif method == 'MICE':
            X_imputed[col] = X_mice[col]
        elif method == 'XGB_NoScale':
            X_imputed[col] = X_xgb_no_scale[col]
        else:  # XGB_Scaled
            X_imputed[col] = X_xgb_scaled[col]
    
    # Predict
    predictions = pipeline['model'].predict(X_imputed)
    probabilities = pipeline['model'].predict_proba(X_imputed)
    
    return predictions, probabilities

# Example usage:
# pipeline = load_best_hybrid_pipeline('covid_results_best_hybrid')
# predictions, probabilities = predict_with_best_hybrid(X_new, pipeline)
"""
    
    with open(os.path.join(output_dir, 'deployment_script.py'), 'w') as f:
        f.write(deployment_script)
    
    print("  ✓ Deployment script saved: deployment_script.py")
    
    print("\n" + "="*80)
    print("✓ ALL MODELS AND ARTIFACTS SAVED SUCCESSFULLY!")
    print("="*80)
    
else:
    print("✗ SAVE CRITERIA NOT MET")
    print("\nCurrent Results:")
    print(f"  AUC: {cv_auc:.4f} (Best previous: {best_prev_auc:.4f})")
    print(f"  Accuracy: {cv_accuracy:.4f} (Best previous: {best_prev_acc:.4f})")
    print(f"  Imputation: {avg_mean_change:.2f}% (Best previous: {best_prev_imp:.2f}%)")
    print("\nResults saved for analysis but models not marked as 'best'.")

# 9. Final summary
print("\n" + "="*80)
print("FINAL SUMMARY - BEST HYBRID IMPUTATION STRATEGY")
print("="*80)

print(f"\n📊 IMPUTATION DETAILS:")
print(f"  Average mean change: {avg_mean_change:.2f}%")
print(f"  Method distribution:")
for method, count in method_counts.items():
    print(f"    • {method}: {count} features ({count/len(feature_names)*100:.1f}%)")

print(f"\n🤖 MODEL PERFORMANCE:")
print(f"  Cross-Validated AUC: {cv_auc:.4f} (+/- {cv_auc_std:.4f})")
print(f"  Cross-Validated Accuracy: {cv_accuracy:.4f} (+/- {cv_accuracy_std:.4f})")
print(f"  Full Dataset AUC: {full_auc:.4f}")
print(f"  Full Dataset Accuracy: {full_accuracy:.4f}")

print(f"\n📈 IMPROVEMENTS:")
print(f"  vs Best AUC: {auc_improvement:+.2f} percentage points")
print(f"  vs Best Accuracy: {acc_improvement:+.2f} percentage points")
print(f"  vs Best Imputation: {imp_improvement:+.2f}%")

print(f"\n💾 OUTPUT LOCATION:")
print(f"  {output_dir}")

print("\n" + "="*80)
print("✓ BEST HYBRID IMPUTATION ANALYSIS COMPLETE!")
print("="*80)
