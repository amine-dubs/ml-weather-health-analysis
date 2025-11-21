"""
Test Outlier Removal Before Model Training
Compare model performance with different outlier detection methods
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
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
output_dir = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\covid_results_outlier_removal'
os.makedirs(output_dir, exist_ok=True)

print("="*80)
print("OUTLIER REMOVAL ANALYSIS")
print("Testing multiple outlier detection methods before model training")
print("="*80)

# 1. Load and prepare data
print("\n1. Loading and preparing dataset...")
data_path = r'c:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\Dataset3.csv'
df_original = pd.read_csv(data_path)

print(f"Original dataset shape: {df_original.shape}")
print(f"Missing values: {df_original.isnull().sum().sum()}")

# Separate features and target
X_with_missing = df_original.drop('SARSCov', axis=1)
y = df_original['SARSCov']

# Apply KNN imputation (best performing method)
print("\nApplying KNN imputation (k=5)...")
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed = pd.DataFrame(
    knn_imputer.fit_transform(X_with_missing),
    columns=X_with_missing.columns,
    index=X_with_missing.index
)

print(f"✓ Imputation completed")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 2. Baseline model (no outlier removal)
print("\n2. Training baseline model (NO outlier removal)...")
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
cv_baseline = cross_validate(xgb_baseline, X_imputed, y, cv=cv, scoring=scoring, 
                             return_train_score=False, n_jobs=-1)

baseline_results = {
    'Method': 'Baseline (No Removal)',
    'Samples': len(X_imputed),
    'Removed': 0,
    'Removed_%': 0.0,
    'CV_AUC': cv_baseline['test_roc_auc'].mean(),
    'CV_AUC_Std': cv_baseline['test_roc_auc'].std(),
    'CV_Accuracy': cv_baseline['test_accuracy'].mean(),
    'CV_Accuracy_Std': cv_baseline['test_accuracy'].std(),
    'CV_Precision': cv_baseline['test_precision'].mean(),
    'CV_Recall': cv_baseline['test_recall'].mean(),
    'CV_F1': cv_baseline['test_f1'].mean()
}

print(f"✓ Baseline Results:")
print(f"  AUC-ROC: {baseline_results['CV_AUC']:.4f} (+/- {baseline_results['CV_AUC_Std']:.4f})")
print(f"  Accuracy: {baseline_results['CV_Accuracy']:.4f} (+/- {baseline_results['CV_Accuracy_Std']:.4f})")
print(f"  Samples: {baseline_results['Samples']}")

# 3. Test outlier detection methods
print("\n3. Testing outlier detection methods...")
print("-"*80)

outlier_methods = {}
all_results = [baseline_results]

# Scale data for outlier detection
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Method 1: Isolation Forest
print("\n[1/4] Isolation Forest...")
for contamination in [0.05, 0.10, 0.15]:
    print(f"  Testing contamination={contamination}...")
    
    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    outlier_labels = iso_forest.fit_predict(X_scaled)
    inlier_mask = outlier_labels == 1
    
    X_clean = X_imputed[inlier_mask]
    y_clean = y[inlier_mask]
    
    # Train model
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    cv_results = cross_validate(xgb_model, X_clean, y_clean, cv=cv, scoring=scoring, 
                                return_train_score=False, n_jobs=-1)
    
    method_name = f'IsolationForest (c={contamination})'
    outlier_methods[method_name] = {
        'mask': inlier_mask,
        'X': X_clean,
        'y': y_clean
    }
    
    result = {
        'Method': method_name,
        'Samples': len(X_clean),
        'Removed': len(X_imputed) - len(X_clean),
        'Removed_%': ((len(X_imputed) - len(X_clean)) / len(X_imputed)) * 100,
        'CV_AUC': cv_results['test_roc_auc'].mean(),
        'CV_AUC_Std': cv_results['test_roc_auc'].std(),
        'CV_Accuracy': cv_results['test_accuracy'].mean(),
        'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
        'CV_Precision': cv_results['test_precision'].mean(),
        'CV_Recall': cv_results['test_recall'].mean(),
        'CV_F1': cv_results['test_f1'].mean()
    }
    all_results.append(result)
    
    print(f"    Removed: {result['Removed']} samples ({result['Removed_%']:.1f}%)")
    print(f"    AUC: {result['CV_AUC']:.4f}, Acc: {result['CV_Accuracy']:.4f}")

# Method 2: Local Outlier Factor
print("\n[2/4] Local Outlier Factor...")
for contamination in [0.05, 0.10, 0.15]:
    print(f"  Testing contamination={contamination}...")
    
    lof = LocalOutlierFactor(
        contamination=contamination,
        n_neighbors=20,
        n_jobs=-1
    )
    outlier_labels = lof.fit_predict(X_scaled)
    inlier_mask = outlier_labels == 1
    
    X_clean = X_imputed[inlier_mask]
    y_clean = y[inlier_mask]
    
    # Train model
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    cv_results = cross_validate(xgb_model, X_clean, y_clean, cv=cv, scoring=scoring, 
                                return_train_score=False, n_jobs=-1)
    
    method_name = f'LocalOutlierFactor (c={contamination})'
    outlier_methods[method_name] = {
        'mask': inlier_mask,
        'X': X_clean,
        'y': y_clean
    }
    
    result = {
        'Method': method_name,
        'Samples': len(X_clean),
        'Removed': len(X_imputed) - len(X_clean),
        'Removed_%': ((len(X_imputed) - len(X_clean)) / len(X_imputed)) * 100,
        'CV_AUC': cv_results['test_roc_auc'].mean(),
        'CV_AUC_Std': cv_results['test_roc_auc'].std(),
        'CV_Accuracy': cv_results['test_accuracy'].mean(),
        'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
        'CV_Precision': cv_results['test_precision'].mean(),
        'CV_Recall': cv_results['test_recall'].mean(),
        'CV_F1': cv_results['test_f1'].mean()
    }
    all_results.append(result)
    
    print(f"    Removed: {result['Removed']} samples ({result['Removed_%']:.1f}%)")
    print(f"    AUC: {result['CV_AUC']:.4f}, Acc: {result['CV_Accuracy']:.4f}")

# Method 3: Elliptic Envelope
print("\n[3/4] Elliptic Envelope (Robust Covariance)...")
for contamination in [0.05, 0.10, 0.15]:
    print(f"  Testing contamination={contamination}...")
    
    try:
        elliptic = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        outlier_labels = elliptic.fit_predict(X_scaled)
        inlier_mask = outlier_labels == 1
        
        X_clean = X_imputed[inlier_mask]
        y_clean = y[inlier_mask]
        
        # Train model
        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        cv_results = cross_validate(xgb_model, X_clean, y_clean, cv=cv, scoring=scoring, 
                                    return_train_score=False, n_jobs=-1)
        
        method_name = f'EllipticEnvelope (c={contamination})'
        outlier_methods[method_name] = {
            'mask': inlier_mask,
            'X': X_clean,
            'y': y_clean
        }
        
        result = {
            'Method': method_name,
            'Samples': len(X_clean),
            'Removed': len(X_imputed) - len(X_clean),
            'Removed_%': ((len(X_imputed) - len(X_clean)) / len(X_imputed)) * 100,
            'CV_AUC': cv_results['test_roc_auc'].mean(),
            'CV_AUC_Std': cv_results['test_roc_auc'].std(),
            'CV_Accuracy': cv_results['test_accuracy'].mean(),
            'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
            'CV_Precision': cv_results['test_precision'].mean(),
            'CV_Recall': cv_results['test_recall'].mean(),
            'CV_F1': cv_results['test_f1'].mean()
        }
        all_results.append(result)
        
        print(f"    Removed: {result['Removed']} samples ({result['Removed_%']:.1f}%)")
        print(f"    AUC: {result['CV_AUC']:.4f}, Acc: {result['CV_Accuracy']:.4f}")
    except Exception as e:
        print(f"    ✗ Failed: {str(e)}")

# Method 4: Statistical (Z-score) method
print("\n[4/4] Statistical Z-score method...")
for threshold in [2.5, 3.0, 3.5]:
    print(f"  Testing threshold={threshold}...")
    
    # Calculate Z-scores
    z_scores = np.abs((X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0))
    outlier_mask = (z_scores < threshold).all(axis=1)
    
    X_clean = X_imputed[outlier_mask]
    y_clean = y[outlier_mask]
    
    # Train model
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    
    cv_results = cross_validate(xgb_model, X_clean, y_clean, cv=cv, scoring=scoring, 
                                return_train_score=False, n_jobs=-1)
    
    method_name = f'Z-score (threshold={threshold})'
    outlier_methods[method_name] = {
        'mask': outlier_mask,
        'X': X_clean,
        'y': y_clean
    }
    
    result = {
        'Method': method_name,
        'Samples': len(X_clean),
        'Removed': len(X_imputed) - len(X_clean),
        'Removed_%': ((len(X_imputed) - len(X_clean)) / len(X_imputed)) * 100,
        'CV_AUC': cv_results['test_roc_auc'].mean(),
        'CV_AUC_Std': cv_results['test_roc_auc'].std(),
        'CV_Accuracy': cv_results['test_accuracy'].mean(),
        'CV_Accuracy_Std': cv_results['test_accuracy'].std(),
        'CV_Precision': cv_results['test_precision'].mean(),
        'CV_Recall': cv_results['test_recall'].mean(),
        'CV_F1': cv_results['test_f1'].mean()
    }
    all_results.append(result)
    
    print(f"    Removed: {result['Removed']} samples ({result['Removed_%']:.1f}%)")
    print(f"    AUC: {result['CV_AUC']:.4f}, Acc: {result['CV_Accuracy']:.4f}")

# 4. Compare all results
print("\n4. Comparing all methods...")
print("="*80)

results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('CV_AUC', ascending=False)

print("\n" + "="*80)
print("COMPREHENSIVE RESULTS")
print("="*80)
print(f"\n{'Method':<35} {'Samples':<10} {'Removed':<10} {'AUC':<10} {'Accuracy':<10}")
print("-"*80)
for _, row in results_df.iterrows():
    print(f"{row['Method']:<35} {row['Samples']:<10} {row['Removed']:<10} "
          f"{row['CV_AUC']:<10.4f} {row['CV_Accuracy']:<10.4f}")

best_method = results_df.iloc[0]
baseline_auc = baseline_results['CV_AUC']
best_auc = best_method['CV_AUC']
auc_improvement = (best_auc - baseline_auc) * 100

print("\n" + "="*80)
print(f"🏆 BEST METHOD: {best_method['Method']}")
print(f"  AUC-ROC: {best_auc:.4f} (+/- {best_method['CV_AUC_Std']:.4f})")
print(f"  Accuracy: {best_method['CV_Accuracy']:.4f} (+/- {best_method['CV_Accuracy_Std']:.4f})")
print(f"  Samples removed: {best_method['Removed']} ({best_method['Removed_%']:.1f}%)")
print(f"  Improvement vs baseline: {auc_improvement:+.2f} pp")
print("="*80)

# Save results
results_df.to_csv(os.path.join(output_dir, 'outlier_removal_comparison.csv'), index=False)
print(f"\n✓ Results saved: {output_dir}/outlier_removal_comparison.csv")

# 5. Visualizations
print("\n5. Generating visualizations...")
print("-"*80)

fig = plt.figure(figsize=(20, 14))

# Plot 1: AUC Comparison
ax1 = plt.subplot(2, 3, 1)
methods_short = [m.split('(')[0].strip() if '(' in m else m for m in results_df['Method']]
colors = ['#e74c3c' if 'Baseline' in m else '#3498db' if best_method['Method'] == results_df.iloc[i]['Method'] 
          else '#95a5a6' for i, m in enumerate(results_df['Method'])]

bars = plt.barh(range(len(results_df)), results_df['CV_AUC'], color=colors, alpha=0.8, edgecolor='black')
plt.yticks(range(len(results_df)), methods_short, fontsize=9)
plt.xlabel('AUC-ROC', fontsize=11, fontweight='bold')
plt.title('Model Performance by Outlier Removal Method', fontsize=13, fontweight='bold')
plt.axvline(x=baseline_auc, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
plt.legend(fontsize=10)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, results_df['CV_AUC'])):
    plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', fontsize=9, fontweight='bold')

# Plot 2: Accuracy Comparison
ax2 = plt.subplot(2, 3, 2)
bars = plt.barh(range(len(results_df)), results_df['CV_Accuracy'], color=colors, alpha=0.8, edgecolor='black')
plt.yticks(range(len(results_df)), methods_short, fontsize=9)
plt.xlabel('Accuracy', fontsize=11, fontweight='bold')
plt.title('Accuracy by Outlier Removal Method', fontsize=13, fontweight='bold')
plt.axvline(x=baseline_results['CV_Accuracy'], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
plt.legend(fontsize=10)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()

# Plot 3: Samples Removed
ax3 = plt.subplot(2, 3, 3)
bars = plt.barh(range(len(results_df)), results_df['Removed_%'], color=colors, alpha=0.8, edgecolor='black')
plt.yticks(range(len(results_df)), methods_short, fontsize=9)
plt.xlabel('Samples Removed (%)', fontsize=11, fontweight='bold')
plt.title('Data Removed by Each Method', fontsize=13, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()

for i, (bar, val) in enumerate(zip(bars, results_df['Removed_%'])):
    if val > 0:
        plt.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')

# Plot 4: AUC vs Samples Removed
ax4 = plt.subplot(2, 3, 4)
scatter_colors = ['red' if 'Baseline' in m else 'green' if best_method['Method'] == results_df.iloc[i]['Method']
                  else 'gray' for i, m in enumerate(results_df['Method'])]

plt.scatter(results_df['Removed_%'], results_df['CV_AUC'], 
           c=scatter_colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)

for i, row in results_df.iterrows():
    label = row['Method'].split('(')[0].strip() if '(' in row['Method'] else row['Method']
    plt.annotate(label, (row['Removed_%'], row['CV_AUC']), 
                fontsize=8, ha='right', va='bottom')

plt.xlabel('Samples Removed (%)', fontsize=11, fontweight='bold')
plt.ylabel('AUC-ROC', fontsize=11, fontweight='bold')
plt.title('Trade-off: Performance vs Data Removal', fontsize=13, fontweight='bold')
plt.axhline(y=baseline_auc, color='red', linestyle='--', linewidth=2, alpha=0.5)
plt.grid(alpha=0.3)

# Plot 5: F1-Score Comparison
ax5 = plt.subplot(2, 3, 5)
bars = plt.barh(range(len(results_df)), results_df['CV_F1'], color=colors, alpha=0.8, edgecolor='black')
plt.yticks(range(len(results_df)), methods_short, fontsize=9)
plt.xlabel('F1-Score', fontsize=11, fontweight='bold')
plt.title('F1-Score by Outlier Removal Method', fontsize=13, fontweight='bold')
plt.axvline(x=baseline_results['CV_F1'], color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
plt.legend(fontsize=10)
plt.grid(axis='x', alpha=0.3)
plt.gca().invert_yaxis()

# Plot 6: Precision vs Recall
ax6 = plt.subplot(2, 3, 6)
plt.scatter(results_df['CV_Recall'], results_df['CV_Precision'], 
           c=scatter_colors, s=200, alpha=0.7, edgecolors='black', linewidth=2)

for i, row in results_df.iterrows():
    label = row['Method'].split('(')[0].strip() if '(' in row['Method'] else row['Method']
    plt.annotate(label, (row['CV_Recall'], row['CV_Precision']), 
                fontsize=8, ha='right', va='bottom')

plt.xlabel('Recall', fontsize=11, fontweight='bold')
plt.ylabel('Precision', fontsize=11, fontweight='bold')
plt.title('Precision vs Recall Trade-off', fontsize=13, fontweight='bold')
plt.axvline(x=baseline_results['CV_Recall'], color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(y=baseline_results['CV_Precision'], color='red', linestyle='--', linewidth=1, alpha=0.5)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'outlier_removal_analysis.png'), dpi=300, bbox_inches='tight')
print(f"✓ Visualizations saved: {output_dir}/outlier_removal_analysis.png")
plt.close()

# 6. Decision: Save if improved
print("\n6. Evaluating save criteria...")
print("="*80)

if best_auc > baseline_auc:
    improvement = (best_auc - baseline_auc) * 100
    print(f"✓ IMPROVEMENT DETECTED!")
    print(f"  AUC improved by {improvement:.2f} percentage points")
    print(f"  Method: {best_method['Method']}")
    print(f"\nSaving best model and configuration...")
    
    # Train final model on best configuration
    best_data = outlier_methods[best_method['Method']]
    final_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    final_model.fit(best_data['X'], best_data['y'])
    
    # Save model
    joblib.dump(final_model, os.path.join(output_dir, 'best_model_with_outlier_removal.joblib'))
    joblib.dump(knn_imputer, os.path.join(output_dir, 'knn_imputer.joblib'))
    
    # Save cleaned dataset
    cleaned_df = best_data['X'].copy()
    cleaned_df['SARSCov'] = best_data['y']
    cleaned_df.to_csv(os.path.join(output_dir, 'dataset_outliers_removed.csv'), index=False)
    
    # Save configuration
    config = {
        'method': best_method['Method'],
        'original_samples': len(X_imputed),
        'cleaned_samples': best_method['Samples'],
        'samples_removed': best_method['Removed'],
        'removal_percentage': best_method['Removed_%'],
        'cv_auc': best_method['CV_AUC'],
        'cv_auc_std': best_method['CV_AUC_Std'],
        'cv_accuracy': best_method['CV_Accuracy'],
        'cv_accuracy_std': best_method['CV_Accuracy_Std'],
        'baseline_auc': baseline_auc,
        'improvement_pp': improvement,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(os.path.join(output_dir, 'best_configuration.txt'), 'w') as f:
        f.write("BEST OUTLIER REMOVAL CONFIGURATION\n")
        f.write("="*80 + "\n\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "="*80 + "\n")
        f.write("\nUSAGE:\n")
        f.write("1. Load imputer: joblib.load('knn_imputer.joblib')\n")
        f.write("2. Apply imputation to new data\n")
        f.write(f"3. Apply outlier detection: {best_method['Method']}\n")
        f.write("4. Load model: joblib.load('best_model_with_outlier_removal.joblib')\n")
        f.write("5. Predict on cleaned data\n")
    
    print(f"\n✓ Model saved: {output_dir}/best_model_with_outlier_removal.joblib")
    print(f"✓ Configuration saved: {output_dir}/best_configuration.txt")
    print(f"✓ Cleaned dataset saved: {output_dir}/dataset_outliers_removed.csv")
    
else:
    print("✗ NO IMPROVEMENT")
    print(f"  Best outlier removal AUC: {best_auc:.4f}")
    print(f"  Baseline AUC: {baseline_auc:.4f}")
    print(f"  Difference: {(best_auc - baseline_auc)*100:+.2f} pp")
    print("\nConclusion: Outlier removal does not improve model performance.")
    print("Baseline model without outlier removal is recommended.")

# 7. Final summary
print("\n" + "="*80)
print("FINAL SUMMARY - OUTLIER REMOVAL ANALYSIS")
print("="*80)

print(f"\n📊 BASELINE (No Removal):")
print(f"  Samples: {baseline_results['Samples']}")
print(f"  AUC-ROC: {baseline_results['CV_AUC']:.4f} (+/- {baseline_results['CV_AUC_Std']:.4f})")
print(f"  Accuracy: {baseline_results['CV_Accuracy']:.4f}")

print(f"\n🔍 OUTLIER DETECTION METHODS TESTED:")
print(f"  • Isolation Forest (3 contamination levels)")
print(f"  • Local Outlier Factor (3 contamination levels)")
print(f"  • Elliptic Envelope (3 contamination levels)")
print(f"  • Z-score Statistical (3 thresholds)")
print(f"  Total configurations: {len(results_df)}")

print(f"\n🏆 BEST CONFIGURATION:")
print(f"  Method: {best_method['Method']}")
print(f"  AUC-ROC: {best_method['CV_AUC']:.4f} (+/- {best_method['CV_AUC_Std']:.4f})")
print(f"  Accuracy: {best_method['CV_Accuracy']:.4f}")
print(f"  Samples removed: {best_method['Removed']} ({best_method['Removed_%']:.1f}%)")
print(f"  vs Baseline: {auc_improvement:+.2f} pp")

if best_auc > baseline_auc:
    print(f"\n✅ RECOMMENDATION: Use {best_method['Method']}")
else:
    print(f"\n❌ RECOMMENDATION: Keep baseline (no outlier removal)")

print("\n💾 Output location:")
print(f"  {output_dir}")

print("\n" + "="*80)
print("✓ OUTLIER REMOVAL ANALYSIS COMPLETE!")
print("="*80)
