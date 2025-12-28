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
print("ANOMALY DETECTION WITH SEPARATE EVALUATION ON POSITIVE AND NEGATIVE CLASSES")
print("="*80)

# Load the data
wine_df = pd.read_csv('wine_quality_merged.csv')

type_counts = wine_df['type'].value_counts()
minority_class = type_counts.idxmin()
majority_class = type_counts.idxmax()

print(f"\nDataset: {len(wine_df)} samples")
print(f"Normal class ({majority_class}): {type_counts[majority_class]} samples")
print(f"Anomaly class ({minority_class}): {type_counts[minority_class]} samples")

# Prepare the data
X = wine_df.drop('type', axis=1)
y = (wine_df['type'] == minority_class).astype(int)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} samples")
print(f"  - Normal: {(y_train==0).sum()}")
print(f"  - Anomaly: {(y_train==1).sum()}")

print(f"\nTest set: {len(X_test)} samples")
print(f"  - Normal: {(y_test==0).sum()}")
print(f"  - Anomaly: {(y_test==1).sum()}")

# Separate test set into positive and negative classes
X_test_negative = X_test[y_test == 0]  # Normal (white wines)
y_test_negative = y_test[y_test == 0]

X_test_positive = X_test[y_test == 1]  # Anomaly (red wines)
y_test_positive = y_test[y_test == 1]

print("\n" + "="*80)
print("SEPARATED TEST SETS FOR EVALUATION")
print("="*80)
print(f"X_test_negative (Normal/{majority_class}): {len(X_test_negative)} samples")
print(f"X_test_positive (Anomaly/{minority_class}): {len(X_test_positive)} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_test_negative_scaled = scaler.transform(X_test_negative)
X_test_positive_scaled = scaler.transform(X_test_positive)

# Apply PCA for visualization
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_negative_pca = pca.transform(X_test_negative_scaled)
X_test_positive_pca = pca.transform(X_test_positive_scaled)

# Calculate contamination rate
contamination_rate = (y_train == 1).sum() / len(y_train)
print(f"\nContamination rate (for training): {contamination_rate:.3f}")

# Define algorithms
algorithms = {
    'Isolation Forest': IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=100),
    'One-Class SVM': OneClassSVM(nu=contamination_rate, kernel='rbf', gamma='auto'),
    'Local Outlier Factor': LocalOutlierFactor(contamination=contamination_rate, novelty=True),
    'Elliptic Envelope': EllipticEnvelope(contamination=contamination_rate, random_state=42),
    'DBSCAN': DBSCAN(eps=0.5, min_samples=5)
}

# Store results
results = {}

print("\n" + "="*80)
print("TRAINING AND EVALUATION")
print("="*80)

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, (name, model) in enumerate(algorithms.items()):
    print(f"\n{'='*80}")
    print(f"Algorithm: {name}")
    print(f"{'='*80}")
    
    # Fit the model on FULL training set (imbalanced)
    print(f"\n1. Training on imbalanced data ({len(X_train)} samples)...")
    
    if name == 'DBSCAN':
        # DBSCAN doesn't support novelty detection
        train_pred = model.fit_predict(X_train_scaled)
        train_pred = (train_pred == -1).astype(int)
        
        # For test data
        test_model = DBSCAN(eps=0.5, min_samples=5)
        test_negative_pred = test_model.fit_predict(X_test_negative_scaled)
        test_negative_pred = (test_negative_pred == -1).astype(int)
        
        test_model_pos = DBSCAN(eps=0.5, min_samples=5)
        test_positive_pred = test_model_pos.fit_predict(X_test_positive_scaled)
        test_positive_pred = (test_positive_pred == -1).astype(int)
        
    elif name == 'Local Outlier Factor':
        model.fit(X_train_scaled)
        train_pred = model.predict(X_train_scaled)
        train_pred = (train_pred == -1).astype(int)
        
        test_negative_pred = model.predict(X_test_negative_scaled)
        test_negative_pred = (test_negative_pred == -1).astype(int)
        
        test_positive_pred = model.predict(X_test_positive_scaled)
        test_positive_pred = (test_positive_pred == -1).astype(int)
        
    else:
        model.fit(X_train_scaled)
        train_pred = model.predict(X_train_scaled)
        train_pred = (train_pred == -1).astype(int)
        
        test_negative_pred = model.predict(X_test_negative_scaled)
        test_negative_pred = (test_negative_pred == -1).astype(int)
        
        test_positive_pred = model.predict(X_test_positive_scaled)
        test_positive_pred = (test_positive_pred == -1).astype(int)
    
    print(f"   Training anomalies detected: {train_pred.sum()}/{len(train_pred)}")
    
    # Evaluate on NEGATIVE class (Normal samples)
    print(f"\n2. Predicting on X_test_negative (Normal/{majority_class} wines only)...")
    
    # For normal samples, we expect prediction = 0 (not anomaly)
    # True Negative Rate = samples correctly identified as normal
    true_negatives = (test_negative_pred == 0).sum()
    false_positives = (test_negative_pred == 1).sum()
    
    tnr = true_negatives / len(test_negative_pred)  # Specificity
    fpr = false_positives / len(test_negative_pred)  # False Positive Rate
    
    print(f"   True Negatives (correctly identified as normal): {true_negatives}/{len(test_negative_pred)}")
    print(f"   False Positives (wrongly identified as anomaly): {false_positives}/{len(test_negative_pred)}")
    print(f"   True Negative Rate (Specificity): {tnr:.4f}")
    print(f"   False Positive Rate: {fpr:.4f}")
    
    # Evaluate on POSITIVE class (Anomaly samples)
    print(f"\n3. Predicting on X_test_positive (Anomaly/{minority_class} wines only)...")
    
    # For anomaly samples, we expect prediction = 1 (anomaly)
    # True Positive Rate = samples correctly identified as anomaly
    true_positives = (test_positive_pred == 1).sum()
    false_negatives = (test_positive_pred == 0).sum()
    
    tpr = true_positives / len(test_positive_pred)  # Sensitivity/Recall
    fnr = false_negatives / len(test_positive_pred)  # False Negative Rate
    
    print(f"   True Positives (correctly identified as anomaly): {true_positives}/{len(test_positive_pred)}")
    print(f"   False Negatives (wrongly identified as normal): {false_negatives}/{len(test_positive_pred)}")
    print(f"   True Positive Rate (Recall/Sensitivity): {tpr:.4f}")
    print(f"   False Negative Rate: {fnr:.4f}")
    
    # Calculate overall metrics
    print(f"\n4. Overall Performance Metrics:")
    
    # Combine predictions
    all_pred = np.concatenate([test_negative_pred, test_positive_pred])
    all_true = np.concatenate([y_test_negative, y_test_positive])
    
    accuracy = accuracy_score(all_true, all_pred)
    precision = precision_score(all_true, all_pred, zero_division=0)
    recall = tpr  # Same as TPR
    f1 = f1_score(all_true, all_pred, zero_division=0)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall (TPR): {recall:.4f}")
    print(f"   Specificity (TNR): {tnr:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': tnr,
        'f1_score': f1,
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr
    }
    
    print(f"\n5. Confusion Matrix:")
    cm = np.array([[true_negatives, false_positives],
                   [false_negatives, true_positives]])
    print(cm)
    print(f"\n   [[TN={true_negatives}, FP={false_positives}],")
    print(f"    [FN={false_negatives}, TP={true_positives}]]")
    
    # Visualize
    ax = axes[idx]
    
    # Plot negative class (normal)
    pred_normal_correctly = test_negative_pred == 0
    pred_normal_wrongly = test_negative_pred == 1
    
    ax.scatter(X_test_negative_pca[pred_normal_correctly, 0], 
               X_test_negative_pca[pred_normal_correctly, 1],
               c='blue', alpha=0.6, s=40, label=f'TN: Correct Normal ({true_negatives})',
               edgecolors='k', linewidth=0.3, marker='o')
    
    ax.scatter(X_test_negative_pca[pred_normal_wrongly, 0],
               X_test_negative_pca[pred_normal_wrongly, 1],
               c='orange', alpha=0.7, s=50, label=f'FP: Wrong Anomaly ({false_positives})',
               edgecolors='red', linewidth=1, marker='x')
    
    # Plot positive class (anomaly)
    pred_anomaly_correctly = test_positive_pred == 1
    pred_anomaly_wrongly = test_positive_pred == 0
    
    ax.scatter(X_test_positive_pca[pred_anomaly_correctly, 0],
               X_test_positive_pca[pred_anomaly_correctly, 1],
               c='red', alpha=0.8, s=50, label=f'TP: Correct Anomaly ({true_positives})',
               edgecolors='k', linewidth=0.5, marker='^')
    
    ax.scatter(X_test_positive_pca[pred_anomaly_wrongly, 0],
               X_test_positive_pca[pred_anomaly_wrongly, 1],
               c='lightgreen', alpha=0.7, s=50, label=f'FN: Missed Anomaly ({false_negatives})',
               edgecolors='green', linewidth=1, marker='s')
    
    ax.set_title(f'{name}\nTPR={tpr:.3f}, TNR={tnr:.3f}, F1={f1:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)

# Remove empty subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('wine_separate_class_evaluation.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*80}")
print("Visualization saved as 'wine_separate_class_evaluation.png'")
print(f"{'='*80}")

# Create summary table
print("\n" + "="*80)
print("PERFORMANCE SUMMARY - SEPARATE CLASS EVALUATION")
print("="*80)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print("\nOverall Metrics:")
print(results_df[['accuracy', 'precision', 'recall', 'specificity', 'f1_score']].to_string())

print("\nDetailed Counts:")
print(results_df[['true_positives', 'false_negatives', 'true_negatives', 'false_positives']].to_string())

print("\nRates:")
print(results_df[['tpr', 'fnr', 'tnr', 'fpr']].to_string())

# Save results
results_df.to_csv('wine_separate_evaluation_results.csv')
print("\nResults saved to 'wine_separate_evaluation_results.csv'")

# Find best algorithms
best_f1 = results_df['f1_score'].idxmax()
best_tpr = results_df['tpr'].idxmax()
best_tnr = results_df['tnr'].idxmax()
best_balanced = results_df[['tpr', 'tnr']].mean(axis=1).idxmax()

print(f"\n{'='*80}")
print("BEST PERFORMING ALGORITHMS")
print(f"{'='*80}")
print(f"Best F1-Score:          {best_f1} ({results_df.loc[best_f1, 'f1_score']:.4f})")
print(f"Best TPR (Recall):      {best_tpr} ({results_df.loc[best_tpr, 'tpr']:.4f})")
print(f"Best TNR (Specificity): {best_tnr} ({results_df.loc[best_tnr, 'tnr']:.4f})")
print(f"Best Balanced (TPR+TNR)/2: {best_balanced} ({results_df.loc[best_balanced, ['tpr', 'tnr']].mean():.4f})")

# Create performance comparison plots
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: TPR vs TNR
ax1 = axes2[0, 0]
x_pos = np.arange(len(results_df))
width = 0.35
ax1.bar(x_pos - width/2, results_df['tpr'], width, label='TPR (Recall)', color='red', alpha=0.7, edgecolor='black')
ax1.bar(x_pos + width/2, results_df['tnr'], width, label='TNR (Specificity)', color='blue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Algorithm')
ax1.set_ylabel('Rate')
ax1.set_title('True Positive Rate vs True Negative Rate', fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df.index, rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 1.1])

# Plot 2: FPR vs FNR
ax2 = axes2[0, 1]
ax2.bar(x_pos - width/2, results_df['fpr'], width, label='FPR (False Positive)', color='orange', alpha=0.7, edgecolor='black')
ax2.bar(x_pos + width/2, results_df['fnr'], width, label='FNR (False Negative)', color='green', alpha=0.7, edgecolor='black')
ax2.set_xlabel('Algorithm')
ax2.set_ylabel('Rate')
ax2.set_title('False Positive Rate vs False Negative Rate', fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(results_df.index, rotation=45, ha='right')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 1.1])

# Plot 3: F1-Score
ax3 = axes2[1, 0]
ax3.bar(results_df.index, results_df['f1_score'], color='purple', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Algorithm')
ax3.set_ylabel('F1-Score')
ax3.set_title('F1-Score Comparison', fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1.1])
for i, v in enumerate(results_df['f1_score']):
    ax3.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

# Plot 4: Accuracy vs Balanced Accuracy
ax4 = axes2[1, 1]
balanced_acc = (results_df['tpr'] + results_df['tnr']) / 2
ax4.bar(x_pos - width/2, results_df['accuracy'], width, label='Accuracy', color='teal', alpha=0.7, edgecolor='black')
ax4.bar(x_pos + width/2, balanced_acc, width, label='Balanced Accuracy', color='navy', alpha=0.7, edgecolor='black')
ax4.set_xlabel('Algorithm')
ax4.set_ylabel('Score')
ax4.set_title('Accuracy vs Balanced Accuracy', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(results_df.index, rotation=45, ha='right')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim([0, 1.1])

plt.tight_layout()
plt.savefig('wine_separate_evaluation_metrics.png', dpi=300, bbox_inches='tight')
print("Performance metrics saved as 'wine_separate_evaluation_metrics.png'")

# ============================================================================
# SAVE BEST MODEL (Elliptic Envelope has best F1-score)
# ============================================================================
print("\n" + "="*80)
print("SAVING BEST MODEL")
print("="*80)

# Re-train the best model (Elliptic Envelope) on full training data
best_model = EllipticEnvelope(contamination=contamination_rate, random_state=42)
best_model.fit(X_train_scaled)

# Get best model metrics
best_metrics = results_df.loc['Elliptic Envelope']

# Save all components needed for prediction
model_package = {
    'model_name': 'Elliptic Envelope',
    'model': best_model,
    'scaler': scaler,
    'pca': pca,
    'feature_names': list(X.columns),
    'contamination_rate': contamination_rate,
    'majority_class': majority_class,
    'minority_class': minority_class,
    'metrics': {
        'accuracy': best_metrics['accuracy'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'specificity': best_metrics['specificity'],
        'f1_score': best_metrics['f1_score'],
        'tpr': best_metrics['tpr'],
        'tnr': best_metrics['tnr'],
        'fpr': best_metrics['fpr'],
        'fnr': best_metrics['fnr'],
        'true_positives': int(best_metrics['true_positives']),
        'false_negatives': int(best_metrics['false_negatives']),
        'true_negatives': int(best_metrics['true_negatives']),
        'false_positives': int(best_metrics['false_positives'])
    }
}

with open('best_anomaly_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"Best model (Elliptic Envelope) saved to 'best_anomaly_model.pkl'")
print(f"Model metrics: F1={model_package['metrics']['f1_score']:.4f}, TPR={model_package['metrics']['tpr']:.4f}, TNR={model_package['metrics']['tnr']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print("\nThis evaluation approach helps understand:")
print("  1. How well the model detects NORMAL samples (TNR/Specificity)")
print("  2. How well the model detects ANOMALY samples (TPR/Recall)")
print("  3. Trade-offs between false positives and false negatives")
print("\nGenerated files:")
print("  1. wine_separate_class_evaluation.png - Visualizations with TP/FP/TN/FN")
print("  2. wine_separate_evaluation_metrics.png - Performance comparison charts")
print("  3. wine_separate_evaluation_results.csv - Detailed results")
print("  4. best_anomaly_model.pkl - Saved best model for predictions")
