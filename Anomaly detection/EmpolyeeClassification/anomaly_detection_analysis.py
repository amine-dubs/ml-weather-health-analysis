import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f"\nTrain data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Check class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION IN TRAINING DATA")
print("="*60)
print(train_df['Attrition'].value_counts())
print("\nClass percentages:")
print(train_df['Attrition'].value_counts(normalize=True) * 100)

print("\n" + "="*60)
print("CLASS DISTRIBUTION IN TEST DATA")
print("="*60)
print(test_df['Attrition'].value_counts())
print("\nClass percentages:")
print(test_df['Attrition'].value_counts(normalize=True) * 100)

# Prepare the data for anomaly detection
def prepare_data(df):
    """Prepare data for anomaly detection by encoding categorical variables"""
    df_processed = df.copy()
    
    # Drop Employee ID as it's not useful for detection
    if 'Employee ID' in df_processed.columns:
        df_processed = df_processed.drop('Employee ID', axis=1)
    
    # Store the target variable
    if 'Attrition' in df_processed.columns:
        y = (df_processed['Attrition'] == 'Left').astype(int)  # 1 for anomaly (Left), 0 for normal (Stayed)
        df_processed = df_processed.drop('Attrition', axis=1)
    else:
        y = None
    
    # Encode categorical variables
    label_encoders = {}
    for column in df_processed.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_processed[column] = le.fit_transform(df_processed[column])
        label_encoders[column] = le
    
    return df_processed, y, label_encoders

print("\n" + "="*60)
print("PREPARING DATA FOR ANOMALY DETECTION")
print("="*60)

X_train, y_train, le_train = prepare_data(train_df)
X_test, y_test, le_test = prepare_data(test_df)

print(f"Features shape: {X_train.shape}")
print(f"Features: {list(X_train.columns)}")

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply PCA for visualization (2 components)
print("\nApplying PCA for 2D visualization...")
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Define anomaly detection algorithms
algorithms = {
    'Isolation Forest': IsolationForest(contamination=0.3, random_state=42, n_estimators=100),
    'One-Class SVM': OneClassSVM(nu=0.3, kernel='rbf', gamma='auto'),
    'Local Outlier Factor': LocalOutlierFactor(contamination=0.3, novelty=True),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.3, random_state=42),
    'DBSCAN': DBSCAN(eps=3, min_samples=5)
}

# Store results
results = {}

print("\n" + "="*60)
print("RUNNING ANOMALY DETECTION ALGORITHMS")
print("="*60)

# Create figure for all visualizations
fig, axes = plt.subplots(3, 2, figsize=(15, 18))
axes = axes.flatten()

for idx, (name, model) in enumerate(algorithms.items()):
    print(f"\n{'='*60}")
    print(f"Algorithm: {name}")
    print(f"{'='*60}")
    
    # Fit the model
    if name == 'DBSCAN':
        # DBSCAN doesn't have fit_predict for new data, so we use it directly
        train_pred = model.fit_predict(X_train_scaled)
        # For DBSCAN, -1 indicates outliers/anomalies
        train_pred = (train_pred == -1).astype(int)
        
        # For test data, we'll use the same model parameters on test data
        test_model = DBSCAN(eps=3, min_samples=5)
        test_pred = test_model.fit_predict(X_test_scaled)
        test_pred = (test_pred == -1).astype(int)
        
    elif name == 'Local Outlier Factor':
        # LOF with novelty=True allows predict on new data
        model.fit(X_train_scaled)
        train_pred = model.predict(X_train_scaled)
        train_pred = (train_pred == -1).astype(int)  # -1 for outliers
        
        test_pred = model.predict(X_test_scaled)
        test_pred = (test_pred == -1).astype(int)
        
    else:
        # Standard fit-predict approach
        model.fit(X_train_scaled)
        train_pred = model.predict(X_train_scaled)
        train_pred = (train_pred == -1).astype(int)  # -1 for outliers in sklearn
        
        test_pred = model.predict(X_test_scaled)
        test_pred = (test_pred == -1).astype(int)
    
    # Calculate metrics on test data
    accuracy = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, zero_division=0)
    recall = recall_score(y_test, test_pred, zero_division=0)
    f1 = f1_score(y_test, test_pred, zero_division=0)
    
    # Store results
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_anomalies': train_pred.sum(),
        'test_anomalies': test_pred.sum()
    }
    
    print(f"Training: Detected {train_pred.sum()} anomalies out of {len(train_pred)} samples")
    print(f"Testing: Detected {test_pred.sum()} anomalies out of {len(test_pred)} samples")
    print(f"\nPerformance Metrics (on Test Data):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, test_pred)
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=['Normal (Stayed)', 'Anomaly (Left)']))
    
    # Visualize results on PCA-reduced data
    ax = axes[idx]
    
    # Plot normal points (predicted as normal)
    normal_idx = test_pred == 0
    ax.scatter(X_test_pca[normal_idx, 0], X_test_pca[normal_idx, 1], 
               c='blue', alpha=0.5, s=20, label='Normal', edgecolors='k', linewidth=0.3)
    
    # Plot detected anomalies in red
    anomaly_idx = test_pred == 1
    ax.scatter(X_test_pca[anomaly_idx, 0], X_test_pca[anomaly_idx, 1], 
               c='red', alpha=0.7, s=30, label='Detected Anomaly', edgecolors='k', linewidth=0.5)
    
    ax.set_title(f'{name}\nF1: {f1:.3f}, Acc: {accuracy:.3f}, Recall: {recall:.3f}', 
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('First Principal Component')
    ax.set_ylabel('Second Principal Component')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove the empty subplot
fig.delaxes(axes[5])

plt.tight_layout()
plt.savefig('anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n{'='*60}")
print("Visualization saved as 'anomaly_detection_comparison.png'")
print(f"{'='*60}")

# Create comparison plot with actual labels
fig2, ax2 = plt.subplots(figsize=(10, 8))

# Plot actual data points
normal_actual = y_test == 0
ax2.scatter(X_test_pca[normal_actual, 0], X_test_pca[normal_actual, 1], 
           c='lightblue', alpha=0.6, s=20, label='Actual Normal (Stayed)', 
           edgecolors='k', linewidth=0.3)

anomaly_actual = y_test == 1
ax2.scatter(X_test_pca[anomaly_actual, 0], X_test_pca[anomaly_actual, 1], 
           c='orange', alpha=0.6, s=30, label='Actual Anomaly (Left)', 
           edgecolors='k', linewidth=0.5, marker='^')

ax2.set_title('Actual Data Distribution (Ground Truth)\nEmployee Attrition', 
             fontsize=14, fontweight='bold')
ax2.set_xlabel('First Principal Component')
ax2.set_ylabel('Second Principal Component')
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('actual_data_distribution.png', dpi=300, bbox_inches='tight')
print("Actual data distribution saved as 'actual_data_distribution.png'")

# Create summary table
print("\n" + "="*60)
print("PERFORMANCE SUMMARY - ALL ALGORITHMS")
print("="*60)

results_df = pd.DataFrame(results).T
results_df = results_df.round(4)
print(results_df.to_string())

# Save results to CSV
results_df.to_csv('anomaly_detection_results.csv')
print("\nResults saved to 'anomaly_detection_results.csv'")

# Find best performing algorithm
best_f1 = results_df['f1_score'].idxmax()
best_accuracy = results_df['accuracy'].idxmax()
best_recall = results_df['recall'].idxmax()

print(f"\n{'='*60}")
print("BEST PERFORMING ALGORITHMS")
print(f"{'='*60}")
print(f"Best F1-Score:  {best_f1} ({results_df.loc[best_f1, 'f1_score']:.4f})")
print(f"Best Accuracy:  {best_accuracy} ({results_df.loc[best_accuracy, 'accuracy']:.4f})")
print(f"Best Recall:    {best_recall} ({results_df.loc[best_recall, 'recall']:.4f})")

# Create bar plot for performance comparison
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, titles)):
    ax = axes3[idx // 2, idx % 2]
    results_df[metric].plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'{title} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(title)
    ax.set_xlabel('Algorithm')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for i, v in enumerate(results_df[metric]):
        ax.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPerformance comparison saved as 'performance_metrics_comparison.png'")

# ============================================================================
# SAVE BEST MODEL (Using Elliptic Envelope - has predict() method)
# ============================================================================
print("\n" + "="*60)
print("SAVING BEST MODEL")
print("="*60)

# Note: DBSCAN has best F1-score but doesn't have predict() method
# Using Elliptic Envelope which has predict() for new data
# Elliptic Envelope has best accuracy (0.4896) among models with predict()

contamination_rate = len(y_train[y_train == 1]) / len(y_train)
best_model = EllipticEnvelope(contamination=contamination_rate, random_state=42)
best_model.fit(X_train_scaled)

# Save all components needed for prediction
model_package = {
    'model_name': 'Elliptic Envelope',
    'model': best_model,
    'scaler': scaler,
    'label_encoders': le_train,
    'feature_names': list(X_train.columns),
    'pca': pca,
    'contamination_rate': contamination_rate,
    'metrics': {
        'accuracy': results_df.loc['Elliptic Envelope', 'accuracy'],
        'precision': results_df.loc['Elliptic Envelope', 'precision'],
        'recall': results_df.loc['Elliptic Envelope', 'recall'],
        'f1_score': results_df.loc['Elliptic Envelope', 'f1_score']
    }
}

with open('best_anomaly_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print(f"Best model (Elliptic Envelope) saved to 'best_anomaly_model.pkl'")
print(f"Model metrics: F1={model_package['metrics']['f1_score']:.4f}, Accuracy={model_package['metrics']['accuracy']:.4f}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)
print("\nGenerated files:")
print("  1. anomaly_detection_comparison.png - Visual comparison of all algorithms")
print("  2. actual_data_distribution.png - Ground truth visualization")
print("  3. performance_metrics_comparison.png - Bar charts of metrics")
print("  4. anomaly_detection_results.csv - Detailed results table")
print("  5. best_anomaly_model.pkl - Saved best model for predictions")
