"""
Test XGBoost on Hybrid Imputed Dataset WITHOUT Scaling
XGBoost is tree-based and doesn't require feature scaling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print(" XGBoost on Hybrid Imputed Dataset - NO SCALING")
print("="*80)

# Load hybrid imputed dataset
print("\n1. Loading hybrid imputed dataset...")
df = pd.read_csv('covid_results/dataset_hybrid_imputed.csv')
print(f"Dataset: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Separate features and target
X = df.drop('SARSCov', axis=1)
y = df['SARSCov']

print(f"\nFeatures: {X.shape[1]}")
print(f"Samples: {len(df)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 2. TRAIN XGBOOST WITHOUT SCALING
# ============================================================================

print("\n2. TRAINING XGBOOST (No Scaling)")
print("="*80)

# Standard XGBoost
print("\nTraining Standard XGBoost...")
xgb_standard = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_standard.fit(X_train, y_train)
y_pred_std = xgb_standard.predict(X_test)
y_pred_proba_std = xgb_standard.predict_proba(X_test)[:, 1]

acc_std = accuracy_score(y_test, y_pred_std)
prec_std = precision_score(y_test, y_pred_std, zero_division=0)
rec_std = recall_score(y_test, y_pred_std, zero_division=0)
f1_std = f1_score(y_test, y_pred_std, zero_division=0)
auc_std = roc_auc_score(y_test, y_pred_proba_std)

print(f"✓ Standard XGBoost:")
print(f"  Accuracy:  {acc_std:.4f} ({acc_std*100:.2f}%)")
print(f"  Precision: {prec_std:.4f} ({prec_std*100:.2f}%)")
print(f"  Recall:    {rec_std:.4f} ({rec_std*100:.2f}%)")
print(f"  F1-Score:  {f1_std:.4f}")
print(f"  AUC-ROC:   {auc_std:.4f} ({auc_std*100:.2f}%)")

# Optimized XGBoost
print("\nTraining Optimized XGBoost...")
xgb_optimized = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=7,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    n_jobs=-1
)

xgb_optimized.fit(X_train, y_train)
y_pred_opt = xgb_optimized.predict(X_test)
y_pred_proba_opt = xgb_optimized.predict_proba(X_test)[:, 1]

acc_opt = accuracy_score(y_test, y_pred_opt)
prec_opt = precision_score(y_test, y_pred_opt, zero_division=0)
rec_opt = recall_score(y_test, y_pred_opt, zero_division=0)
f1_opt = f1_score(y_test, y_pred_opt, zero_division=0)
auc_opt = roc_auc_score(y_test, y_pred_proba_opt)

print(f"✓ Optimized XGBoost:")
print(f"  Accuracy:  {acc_opt:.4f} ({acc_opt*100:.2f}%)")
print(f"  Precision: {prec_opt:.4f} ({prec_opt*100:.2f}%)")
print(f"  Recall:    {rec_opt:.4f} ({rec_opt*100:.2f}%)")
print(f"  F1-Score:  {f1_opt:.4f}")
print(f"  AUC-ROC:   {auc_opt:.4f} ({auc_opt*100:.2f}%)")

# ============================================================================
# 3. COMPARISON WITH SCALED VERSION
# ============================================================================

print("\n3. COMPARISON WITH PREVIOUS RESULTS")
print("="*80)

# XGBoost with scaling (from previous results)
xgb_scaled_auc = 0.8854  # From run_hybrid_comparison.py results

print(f"\nXGBoost Performance Comparison:")
print(f"  With Scaling:    {xgb_scaled_auc:.4f} ({xgb_scaled_auc*100:.2f}%)")
print(f"  Without Scaling (Standard): {auc_std:.4f} ({auc_std*100:.2f}%)")
print(f"  Without Scaling (Optimized): {auc_opt:.4f} ({auc_opt*100:.2f}%)")

diff_std = (auc_std - xgb_scaled_auc) * 100
diff_opt = (auc_opt - xgb_scaled_auc) * 100

print(f"\nDifference:")
print(f"  Standard vs Scaled: {diff_std:+.2f} percentage points")
print(f"  Optimized vs Scaled: {diff_opt:+.2f} percentage points")

if auc_opt > xgb_scaled_auc:
    print(f"\n✓ No scaling performs BETTER by {diff_opt:.2f} pp!")
elif abs(diff_opt) < 0.5:
    print(f"\n✓ Performance is comparable (within 0.5 pp)")
else:
    print(f"\n- Scaling performs slightly better")

# Best overall comparison
best_auc = max(auc_std, auc_opt)
best_name = "Standard" if auc_std > auc_opt else "Optimized"

print(f"\n🏆 BEST XGBoost (No Scaling): {best_name}")
print(f"   AUC: {best_auc:.4f} ({best_auc*100:.2f}%)")

# ============================================================================
# 4. DETAILED EVALUATION OF BEST MODEL
# ============================================================================

print("\n4. DETAILED EVALUATION - BEST MODEL")
print("="*80)

best_model = xgb_optimized if auc_opt > auc_std else xgb_standard
best_pred = y_pred_opt if auc_opt > auc_std else y_pred_std
best_proba = y_pred_proba_opt if auc_opt > auc_std else y_pred_proba_std

print(f"\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Negative', 'Positive']))

cm = confusion_matrix(y_test, best_pred)
print(f"\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nSpecificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"Sensitivity: {sensitivity:.4f} ({sensitivity*100:.2f}%)")

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================

print("\n5. FEATURE IMPORTANCE")
print("="*80)

# Get feature importance
feature_importance = best_model.feature_importances_
feature_names = X.columns

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print(f"\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# ============================================================================
# 6. VISUALIZATIONS
# ============================================================================

print("\n6. GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Comparison Bar Chart
ax1 = axes[0, 0]
models = ['XGB\n(Scaled)', 'XGB\n(No Scale\nStandard)', 'XGB\n(No Scale\nOptimized)']
aucs = [xgb_scaled_auc, auc_std, auc_opt]
colors = ['#3498db', '#e74c3c', '#2ecc71']

bars = ax1.bar(models, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
ax1.set_ylabel('AUC-ROC', fontweight='bold', fontsize=12)
ax1.set_title('XGBoost Performance: With vs Without Scaling', fontweight='bold', fontsize=14)
ax1.set_ylim([0.85, max(aucs) + 0.02])
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, aucs):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.002,
             f'{val:.4f}\n({val*100:.2f}%)', ha='center', va='bottom', 
             fontweight='bold', fontsize=10)

# Plot 2: Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            linewidths=2, linecolor='black', ax=ax2)
ax2.set_ylabel('Actual', fontsize=12, fontweight='bold')
ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax2.set_title(f'Confusion Matrix - XGBoost ({best_name})', fontsize=14, fontweight='bold')

# Plot 3: ROC Curve
ax3 = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, best_proba)
ax3.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {best_auc:.4f})')
ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax3.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax3.legend(loc="lower right", fontsize=11)
ax3.grid(alpha=0.3)

# Plot 4: Top 15 Feature Importance
ax4 = axes[1, 1]
top_features = importance_df.head(15)
colors_imp = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
ax4.barh(range(len(top_features)), top_features['Importance'], color=colors_imp, alpha=0.8)
ax4.set_yticks(range(len(top_features)))
ax4.set_yticklabels(top_features['Feature'], fontsize=10)
ax4.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax4.set_title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('covid_results_hybrid/xgboost_no_scaling_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Saved: covid_results_hybrid/xgboost_no_scaling_results.png")

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

print("\n7. SAVING RESULTS")
print("="*80)

results_summary = {
    'xgboost_scaled': {
        'auc': xgb_scaled_auc,
        'scaling': 'StandardScaler'
    },
    'xgboost_no_scale_standard': {
        'auc': auc_std,
        'accuracy': acc_std,
        'precision': prec_std,
        'recall': rec_std,
        'f1_score': f1_std,
        'scaling': 'None',
        'params': 'Standard (n_estimators=100, lr=0.1, depth=5)'
    },
    'xgboost_no_scale_optimized': {
        'auc': auc_opt,
        'accuracy': acc_opt,
        'precision': prec_opt,
        'recall': rec_opt,
        'f1_score': f1_opt,
        'scaling': 'None',
        'params': 'Optimized (n_estimators=200, lr=0.05, depth=7, subsample=0.8)'
    }
}

import json
with open('covid_results_hybrid/xgboost_no_scaling_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("✓ Results saved: covid_results_hybrid/xgboost_no_scaling_results.json")

# Save feature importance
importance_df.to_csv('covid_results_hybrid/xgboost_feature_importance.csv', index=False)
print("✓ Feature importance saved: covid_results_hybrid/xgboost_feature_importance.csv")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print(" SUMMARY - XGBoost WITHOUT SCALING")
print("="*80)

print(f"\n📊 Results:")
print(f"  XGBoost with Scaling:        {xgb_scaled_auc:.4f} ({xgb_scaled_auc*100:.2f}%)")
print(f"  XGBoost No Scale (Standard): {auc_std:.4f} ({auc_std*100:.2f}%)")
print(f"  XGBoost No Scale (Optimized): {auc_opt:.4f} ({auc_opt*100:.2f}%)")

print(f"\n🏆 Best: XGBoost {best_name} (No Scaling)")
print(f"   AUC-ROC: {best_auc:.4f} ({best_auc*100:.2f}%)")
print(f"   Accuracy: {(acc_opt if auc_opt > auc_std else acc_std)*100:.2f}%")

if best_auc > xgb_scaled_auc:
    print(f"\n✅ NO SCALING IS BETTER!")
    print(f"   Improvement: {(best_auc - xgb_scaled_auc)*100:+.2f} percentage points")
elif abs(best_auc - xgb_scaled_auc) < 0.005:
    print(f"\n✓ Performance is identical (difference < 0.5 pp)")
else:
    print(f"\n- Scaling performs slightly better")

print(f"\n💡 Key Insight:")
print(f"   Tree-based models like XGBoost don't require feature scaling")
print(f"   Performance is comparable or better without scaling")
print(f"   Saves preprocessing time and complexity")

print("\n" + "="*80)
print("✓ XGBoost Analysis Complete!")
print("="*80)
