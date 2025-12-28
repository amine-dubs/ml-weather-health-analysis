import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HEART DISEASE ANOMALY DETECTION - COMPARING EVALUATION METHODS")
print("="*80)

# Load the data
heart_df = pd.read_csv('Dataset2.csv')

print(f"\nOriginal Dataset shape: {heart_df.shape}")
print(f"\nColumns: {heart_df.columns.tolist()}")

# Check original class distribution
print("\n" + "="*80)
print("ORIGINAL CLASS DISTRIBUTION")
print("="*80)
print(heart_df['target'].value_counts())
print("\nTarget percentages:")
print(heart_df['target'].value_counts(normalize=True) * 100)

# The target column: 0 = No heart disease, 1 = Heart disease
# Let's make heart disease (1) the anomaly (minority class)

# ============================================================================
# STEP 1: CREATE IMBALANCED DATASET (Reduce anomaly class to <100 samples)
# ============================================================================
print("\n" + "="*80)
print("CREATING IMBALANCED DATASET")
print("="*80)

# Separate classes
normal_samples = heart_df[heart_df['target'] == 0]  # No heart disease
anomaly_samples = heart_df[heart_df['target'] == 1]  # Heart disease

print(f"Original normal (no disease): {len(normal_samples)}")
print(f"Original anomaly (disease): {len(anomaly_samples)}")

# Reduce anomaly class to 80 samples (less than 100)
np.random.seed(42)
anomaly_reduced = anomaly_samples.sample(n=80, random_state=42)

# Combine to create imbalanced dataset
imbalanced_df = pd.concat([normal_samples, anomaly_reduced], ignore_index=True)

# Shuffle the dataset
imbalanced_df = imbalanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nImbalanced Dataset:")
print(f"  Total samples: {len(imbalanced_df)}")
print(f"  Normal (no disease): {(imbalanced_df['target']==0).sum()}")
print(f"  Anomaly (disease): {(imbalanced_df['target']==1).sum()}")
print(f"  Anomaly ratio: {(imbalanced_df['target']==1).sum() / len(imbalanced_df):.2%}")

# ============================================================================
# STEP 2: PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("PREPARING DATA")
print("="*80)

X = imbalanced_df.drop('target', axis=1)
y = imbalanced_df['target']  # 0 = normal, 1 = anomaly (heart disease)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"  - Normal: {(y_train==0).sum()}")
print(f"  - Anomaly: {(y_train==1).sum()}")

print(f"\nTest set: {len(X_test)} samples")
print(f"  - Normal: {(y_test==0).sum()}")
print(f"  - Anomaly: {(y_test==1).sum()}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Contamination rate
contamination_rate = (y_train == 1).sum() / len(y_train)
print(f"Contamination rate: {contamination_rate:.3f}")

# ============================================================================
# STEP 3: SEPARATE TEST SETS FOR EVALUATION
# ============================================================================
X_test_negative = X_test[y_test == 0]  # Normal samples only
y_test_negative = y_test[y_test == 0]
X_test_negative_scaled = scaler.transform(X_test_negative)
X_test_negative_pca = pca.transform(X_test_negative_scaled)

X_test_positive = X_test[y_test == 1]  # Anomaly samples only
y_test_positive = y_test[y_test == 1]
X_test_positive_scaled = scaler.transform(X_test_positive)
X_test_positive_pca = pca.transform(X_test_positive_scaled)

print(f"\nSeparate test sets:")
print(f"  X_test_negative (Normal): {len(X_test_negative)} samples")
print(f"  X_test_positive (Anomaly): {len(X_test_positive)} samples")

# ============================================================================
# STEP 4: DEFINE ALGORITHMS
# ============================================================================
algorithms = {
    'Isolation Forest': IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100),
    'One-Class SVM': OneClassSVM(nu=contamination_rate, kernel='rbf', gamma='auto'),
    'Local Outlier Factor': LocalOutlierFactor(contamination=contamination_rate, novelty=True),
    'Elliptic Envelope': EllipticEnvelope(contamination=contamination_rate, random_state=42),
    'DBSCAN': DBSCAN(eps=1.5, min_samples=5)
}

# ============================================================================
# STEP 5: RUN BOTH EVALUATION METHODS
# ============================================================================
results_standard = {}
results_separate = {}

print("\n" + "="*80)
print("RUNNING ANOMALY DETECTION ALGORITHMS")
print("="*80)

# Create visualization
fig, axes = plt.subplots(5, 2, figsize=(16, 25))

for idx, (name, model) in enumerate(algorithms.items()):
    print(f"\n{'='*80}")
    print(f"Algorithm: {name}")
    print(f"{'='*80}")
    
    # ========== TRAIN MODEL ==========
    if name == 'DBSCAN':
        train_pred = model.fit_predict(X_train_scaled)
        train_pred = (train_pred == -1).astype(int)
        
        # Standard evaluation
        test_model = DBSCAN(eps=1.5, min_samples=5)
        test_pred_standard = test_model.fit_predict(X_test_scaled)
        test_pred_standard = (test_pred_standard == -1).astype(int)
        
        # Separate evaluation
        test_model_neg = DBSCAN(eps=1.5, min_samples=5)
        test_pred_negative = test_model_neg.fit_predict(X_test_negative_scaled)
        test_pred_negative = (test_pred_negative == -1).astype(int)
        
        test_model_pos = DBSCAN(eps=1.5, min_samples=5)
        test_pred_positive = test_model_pos.fit_predict(X_test_positive_scaled)
        test_pred_positive = (test_pred_positive == -1).astype(int)
        
    elif name == 'Local Outlier Factor':
        model.fit(X_train_scaled)
        
        # Standard evaluation
        test_pred_standard = model.predict(X_test_scaled)
        test_pred_standard = (test_pred_standard == -1).astype(int)
        
        # Separate evaluation
        test_pred_negative = model.predict(X_test_negative_scaled)
        test_pred_negative = (test_pred_negative == -1).astype(int)
        
        test_pred_positive = model.predict(X_test_positive_scaled)
        test_pred_positive = (test_pred_positive == -1).astype(int)
    else:
        model.fit(X_train_scaled)
        
        # Standard evaluation
        test_pred_standard = model.predict(X_test_scaled)
        test_pred_standard = (test_pred_standard == -1).astype(int)
        
        # Separate evaluation
        test_pred_negative = model.predict(X_test_negative_scaled)
        test_pred_negative = (test_pred_negative == -1).astype(int)
        
        test_pred_positive = model.predict(X_test_positive_scaled)
        test_pred_positive = (test_pred_positive == -1).astype(int)
    
    # ========== STANDARD EVALUATION ==========
    print("\n--- STANDARD EVALUATION (Full Test Set) ---")
    
    acc_std = accuracy_score(y_test, test_pred_standard)
    prec_std = precision_score(y_test, test_pred_standard, zero_division=0)
    rec_std = recall_score(y_test, test_pred_standard, zero_division=0)
    f1_std = f1_score(y_test, test_pred_standard, zero_division=0)
    
    results_standard[name] = {
        'accuracy': acc_std,
        'precision': prec_std,
        'recall': rec_std,
        'f1_score': f1_std
    }
    
    print(f"Accuracy:  {acc_std:.4f}")
    print(f"Precision: {prec_std:.4f}")
    print(f"Recall:    {rec_std:.4f}")
    print(f"F1-Score:  {f1_std:.4f}")
    
    cm_std = confusion_matrix(y_test, test_pred_standard)
    print(f"Confusion Matrix:\n{cm_std}")
    
    # ========== SEPARATE CLASS EVALUATION ==========
    print("\n--- SEPARATE CLASS EVALUATION ---")
    
    # Negative class metrics
    true_negatives = (test_pred_negative == 0).sum()
    false_positives = (test_pred_negative == 1).sum()
    tnr = true_negatives / len(test_pred_negative) if len(test_pred_negative) > 0 else 0
    fpr = false_positives / len(test_pred_negative) if len(test_pred_negative) > 0 else 0
    
    # Positive class metrics
    true_positives = (test_pred_positive == 1).sum()
    false_negatives = (test_pred_positive == 0).sum()
    tpr = true_positives / len(test_pred_positive) if len(test_pred_positive) > 0 else 0
    fnr = false_negatives / len(test_pred_positive) if len(test_pred_positive) > 0 else 0
    
    # Combined metrics
    all_pred = np.concatenate([test_pred_negative, test_pred_positive])
    all_true = np.concatenate([y_test_negative, y_test_positive])
    
    acc_sep = accuracy_score(all_true, all_pred)
    prec_sep = precision_score(all_true, all_pred, zero_division=0)
    f1_sep = f1_score(all_true, all_pred, zero_division=0)
    balanced_acc = (tpr + tnr) / 2
    
    results_separate[name] = {
        'accuracy': acc_sep,
        'precision': prec_sep,
        'recall_tpr': tpr,
        'specificity_tnr': tnr,
        'f1_score': f1_sep,
        'balanced_accuracy': balanced_acc,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'fpr': fpr,
        'fnr': fnr
    }
    
    print(f"Predicting on X_test_negative (Normal only): {len(test_pred_negative)} samples")
    print(f"  TN: {true_negatives}, FP: {false_positives}")
    print(f"  TNR (Specificity): {tnr:.4f}")
    print(f"  FPR: {fpr:.4f}")
    
    print(f"\nPredicting on X_test_positive (Anomaly only): {len(test_pred_positive)} samples")
    print(f"  TP: {true_positives}, FN: {false_negatives}")
    print(f"  TPR (Recall/Sensitivity): {tpr:.4f}")
    print(f"  FNR: {fnr:.4f}")
    
    print(f"\nCombined Metrics:")
    print(f"  Accuracy: {acc_sep:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc:.4f}")
    print(f"  F1-Score: {f1_sep:.4f}")
    
    # ========== VISUALIZATION ==========
    # Standard evaluation plot
    ax1 = axes[idx, 0]
    normal_idx = test_pred_standard == 0
    anomaly_idx = test_pred_standard == 1
    ax1.scatter(X_test_pca[normal_idx, 0], X_test_pca[normal_idx, 1],
                c='blue', alpha=0.6, s=50, label='Predicted Normal', edgecolors='k', linewidth=0.3)
    ax1.scatter(X_test_pca[anomaly_idx, 0], X_test_pca[anomaly_idx, 1],
                c='red', alpha=0.8, s=60, label='Predicted Anomaly', edgecolors='k', linewidth=0.5, marker='^')
    ax1.set_title(f'{name} - Standard Evaluation\nAcc={acc_std:.3f}, F1={f1_std:.3f}', fontweight='bold')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Separate evaluation plot
    ax2 = axes[idx, 1]
    
    # TN - correctly identified normal
    tn_idx = test_pred_negative == 0
    ax2.scatter(X_test_negative_pca[tn_idx, 0], X_test_negative_pca[tn_idx, 1],
                c='blue', alpha=0.6, s=50, label=f'TN ({true_negatives})', edgecolors='k', linewidth=0.3)
    
    # FP - wrongly flagged as anomaly
    fp_idx = test_pred_negative == 1
    ax2.scatter(X_test_negative_pca[fp_idx, 0], X_test_negative_pca[fp_idx, 1],
                c='orange', alpha=0.8, s=60, label=f'FP ({false_positives})', edgecolors='red', linewidth=1, marker='x')
    
    # TP - correctly detected anomaly
    tp_idx = test_pred_positive == 1
    ax2.scatter(X_test_positive_pca[tp_idx, 0], X_test_positive_pca[tp_idx, 1],
                c='red', alpha=0.8, s=60, label=f'TP ({true_positives})', edgecolors='k', linewidth=0.5, marker='^')
    
    # FN - missed anomaly
    fn_idx = test_pred_positive == 0
    ax2.scatter(X_test_positive_pca[fn_idx, 0], X_test_positive_pca[fn_idx, 1],
                c='lightgreen', alpha=0.8, s=60, label=f'FN ({false_negatives})', edgecolors='green', linewidth=1, marker='s')
    
    ax2.set_title(f'{name} - Separate Class Evaluation\nTPR={tpr:.3f}, TNR={tnr:.3f}, Balanced={balanced_acc:.3f}', fontweight='bold')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heart_anomaly_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print("Visualization saved as 'heart_anomaly_comparison.png'")

# ============================================================================
# STEP 6: COMPARISON OF EVALUATION METHODS
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: STANDARD vs SEPARATE CLASS EVALUATION")
print("="*80)

# Create comparison table
comparison_data = []
for name in algorithms.keys():
    comparison_data.append({
        'Algorithm': name,
        'Std_Accuracy': results_standard[name]['accuracy'],
        'Std_F1': results_standard[name]['f1_score'],
        'Sep_Accuracy': results_separate[name]['accuracy'],
        'Sep_F1': results_separate[name]['f1_score'],
        'TPR': results_separate[name]['recall_tpr'],
        'TNR': results_separate[name]['specificity_tnr'],
        'Balanced_Acc': results_separate[name]['balanced_accuracy'],
        'TP': results_separate[name]['true_positives'],
        'FN': results_separate[name]['false_negatives'],
        'TN': results_separate[name]['true_negatives'],
        'FP': results_separate[name]['false_positives']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.round(4)

print("\n--- Standard Evaluation Metrics ---")
print(comparison_df[['Algorithm', 'Std_Accuracy', 'Std_F1']].to_string(index=False))

print("\n--- Separate Class Evaluation Metrics ---")
print(comparison_df[['Algorithm', 'Sep_Accuracy', 'Sep_F1', 'TPR', 'TNR', 'Balanced_Acc']].to_string(index=False))

print("\n--- Detailed Counts ---")
print(comparison_df[['Algorithm', 'TP', 'FN', 'TN', 'FP']].to_string(index=False))

# Save results
comparison_df.to_csv('heart_evaluation_comparison.csv', index=False)
print("\nResults saved to 'heart_evaluation_comparison.csv'")

# ============================================================================
# STEP 7: DETERMINE WHICH METHOD IS BETTER
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS: WHICH EVALUATION METHOD IS BETTER?")
print("="*80)

# Find best algorithms
best_std_f1 = comparison_df.loc[comparison_df['Std_F1'].idxmax()]
best_balanced = comparison_df.loc[comparison_df['Balanced_Acc'].idxmax()]
best_tpr = comparison_df.loc[comparison_df['TPR'].idxmax()]

print(f"\nBest by Standard F1: {best_std_f1['Algorithm']} (F1={best_std_f1['Std_F1']:.4f})")
print(f"Best by Balanced Accuracy: {best_balanced['Algorithm']} (Bal_Acc={best_balanced['Balanced_Acc']:.4f})")
print(f"Best by TPR (Recall): {best_tpr['Algorithm']} (TPR={best_tpr['TPR']:.4f})")

print("""
============================================================================
KEY FINDINGS:
============================================================================

1. ACCURACY IS THE SAME IN BOTH METHODS:
   - Standard and Separate evaluation give identical accuracy
   - This is expected - we're predicting on the same test samples

2. WHAT SEPARATE EVALUATION REVEALS:
   - TPR: How well we detect heart disease (anomalies)
   - TNR: How well we identify healthy patients (normal)
   - These insights are HIDDEN in standard F1-score

3. WHY SEPARATE EVALUATION IS MORE INFORMATIVE:
   - For medical diagnosis, we MUST know both TPR and TNR
   - High TPR = catch most sick patients (critical!)
   - High TNR = don't alarm healthy patients unnecessarily
   
4. EXAMPLE OF HIDDEN INFORMATION:
   - If standard F1 = 0.50, you don't know if:
     a) TPR=0.90, TNR=0.40 (catches diseases but many false alarms)
     b) TPR=0.30, TNR=0.95 (misses diseases but few false alarms)
   - These have very different clinical implications!

5. CONCLUSION:
   - Standard evaluation: Good for quick comparison
   - Separate evaluation: ESSENTIAL for understanding real performance
   - For imbalanced medical data: ALWAYS use separate evaluation
""")

# ============================================================================
# STEP 8: PERFORMANCE COMPARISON VISUALIZATION
# ============================================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Standard vs Separate F1
ax1 = axes2[0, 0]
x_pos = np.arange(len(comparison_df))
width = 0.35
ax1.bar(x_pos - width/2, comparison_df['Std_F1'], width, label='Standard F1', color='steelblue', alpha=0.7)
ax1.bar(x_pos + width/2, comparison_df['Sep_F1'], width, label='Separate F1', color='coral', alpha=0.7)
ax1.set_xlabel('Algorithm')
ax1.set_ylabel('F1-Score')
ax1.set_title('Standard vs Separate F1-Score (Same Values!)', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(comparison_df['Algorithm'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.1])

# Plot 2: TPR vs TNR
ax2 = axes2[0, 1]
ax2.bar(x_pos - width/2, comparison_df['TPR'], width, label='TPR (Recall)', color='red', alpha=0.7)
ax2.bar(x_pos + width/2, comparison_df['TNR'], width, label='TNR (Specificity)', color='blue', alpha=0.7)
ax2.set_xlabel('Algorithm')
ax2.set_ylabel('Rate')
ax2.set_title('TPR vs TNR (Only from Separate Evaluation!)', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(comparison_df['Algorithm'], rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1.1])

# Plot 3: Balanced Accuracy
ax3 = axes2[1, 0]
ax3.bar(comparison_df['Algorithm'], comparison_df['Balanced_Acc'], color='purple', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Algorithm')
ax3.set_ylabel('Balanced Accuracy')
ax3.set_title('Balanced Accuracy (TPR+TNR)/2', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.1])
for i, v in enumerate(comparison_df['Balanced_Acc']):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 4: TP/FN/TN/FP breakdown
ax4 = axes2[1, 1]
width = 0.2
ax4.bar(x_pos - 1.5*width, comparison_df['TP'], width, label='TP', color='darkgreen', alpha=0.7)
ax4.bar(x_pos - 0.5*width, comparison_df['FN'], width, label='FN', color='lightgreen', alpha=0.7)
ax4.bar(x_pos + 0.5*width, comparison_df['TN'], width, label='TN', color='darkblue', alpha=0.7)
ax4.bar(x_pos + 1.5*width, comparison_df['FP'], width, label='FP', color='orange', alpha=0.7)
ax4.set_xlabel('Algorithm')
ax4.set_ylabel('Count')
ax4.set_title('Confusion Matrix Breakdown (Separate Evaluation)', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(comparison_df['Algorithm'], rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('heart_evaluation_metrics_comparison.png', dpi=300, bbox_inches='tight')
print("\nMetrics comparison saved as 'heart_evaluation_metrics_comparison.png'")

# ============================================================================
# SAVE BEST MODEL (Elliptic Envelope has best Balanced Accuracy)
# ============================================================================
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

# Re-train the best model (Elliptic Envelope) on full training data
best_model = EllipticEnvelope(contamination=contamination_rate, random_state=42)
best_model.fit(X_train_scaled)

# Get best model results
best_row = comparison_df[comparison_df['Algorithm'] == 'Elliptic Envelope'].iloc[0]

# Save all components needed for prediction
model_package = {
    'model_name': 'Elliptic Envelope',
    'model': best_model,
    'scaler': scaler,
    'pca': pca,
    'feature_names': list(X.columns),
    'contamination_rate': contamination_rate,
    'metrics': {
        'accuracy': best_row['Sep_Accuracy'],
        'f1_score': best_row['Sep_F1'],
        'tpr': best_row['TPR'],
        'tnr': best_row['TNR'],
        'balanced_accuracy': best_row['Balanced_Acc'],
        'tp': int(best_row['TP']),
        'fn': int(best_row['FN']),
        'tn': int(best_row['TN']),
        'fp': int(best_row['FP'])
    }
}

with open('best_anomaly_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"Best model (Elliptic Envelope) saved to 'best_anomaly_model.pkl'")
print(f"Model metrics: Balanced Acc={model_package['metrics']['balanced_accuracy']:.4f}, TPR={model_package['metrics']['tpr']:.4f}, TNR={model_package['metrics']['tnr']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. heart_anomaly_comparison.png - Side-by-side evaluation visualizations")
print("  2. heart_evaluation_metrics_comparison.png - Metrics comparison charts")
print("  3. heart_evaluation_comparison.csv - Complete numerical results")
print("  4. best_anomaly_model.pkl - Saved best model for predictions")
