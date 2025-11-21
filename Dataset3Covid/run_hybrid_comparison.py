"""
COVID-19 Analysis with Hybrid Imputation
Uses pre-computed hybrid imputed dataset (0.72% avg mean change)
Compares results with original KNN-only imputation
"""

import pandas as pd
import numpy as np
import os
import json
import warnings
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

warnings.filterwarnings('ignore')

print("="*80)
print(" COVID-19 ANALYSIS WITH HYBRID IMPUTATION")
print(" Hybrid Strategy: KNN (5 features) + MICE (27 features)")
print(" Imputation Quality: 0.72% avg mean change")
print("="*80)

# Create output directories
output_dir = 'covid_results_hybrid'
models_dir = os.path.join(output_dir, 'models')
plots_dir = os.path.join(output_dir, 'plots')

for dir_path in [output_dir, models_dir, plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 1. LOAD HYBRID IMPUTED DATASET
# ============================================================================

print("\\n1. LOADING HYBRID IMPUTED DATASET")
print("="*80)

# Load the pre-computed hybrid imputed dataset
df = pd.read_csv('covid_results/dataset_hybrid_imputed.csv')
print(f"Dataset loaded: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()} (should be 0)")

# Separate features and target
X = df.drop('SARSCov', axis=1)
y = df['SARSCov']

print(f"\\nFeatures: {X.shape[1]}")
print(f"Samples: {len(df)}")
print(f"\\nTarget distribution:")
print(y.value_counts())

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================

print("\\n2. TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================================================
# 3. SCALE FEATURES
# ============================================================================

print("\\n3. FEATURE SCALING")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✓ StandardScaler applied")

# ============================================================================
# 4. TRAIN MODELS
# ============================================================================

print("\\n4. TRAINING MODELS")
print("="*80)

models = {
    # Linear Models
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    
    # K-Nearest Neighbors
    'KNN (k=3)': KNeighborsClassifier(n_neighbors=3),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    
    # Support Vector Machines with different kernels
    'SVM (Linear)': SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced'),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'SVM (Polynomial)': SVC(kernel='poly', degree=3, probability=True, random_state=42, class_weight='balanced'),
    'SVM (Sigmoid)': SVC(kernel='sigmoid', probability=True, random_state=42, class_weight='balanced'),
    
    # Tree-based models
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, 
                                           class_weight='balanced', n_jobs=-1),
    
    # Boosting models
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                                     max_depth=5, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    
    # Naive Bayes
    'Naive Bayes': GaussianNB()
}

if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                          random_state=42, use_label_encoder=False,
                                          eval_metric='logloss', n_jobs=-1)

if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                            random_state=42, n_jobs=-1, verbose=-1)

results = []

for name, model in models.items():
    print(f"\\nTraining {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'AUC-ROC': auc,
        'Model_Object': model
    })
    
    auc_str = f"{auc:.4f}" if auc is not None else "N/A"
    print(f"✓ AUC: {auc_str}, Acc: {acc:.4f}")

results_df = pd.DataFrame(results).sort_values('AUC-ROC', ascending=False)

print("\\n" + "="*80)
print("BASE MODELS RESULTS")
print("="*80)
print(results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_string(index=False))

# ============================================================================
# 5. TRAIN ENSEMBLE MODELS
# ============================================================================

print("\\n5. TRAINING ENSEMBLE MODELS")
print("="*80)

# Base estimators for ensembles
base_estimators = [
    ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')),
    ('random_forest', RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42,
                                             class_weight='balanced', n_jobs=-1)),
    ('gradient_boost', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                   max_depth=5, random_state=42)),
]

if XGBOOST_AVAILABLE:
    base_estimators.append(('xgboost', xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                                          max_depth=5, random_state=42,
                                                          use_label_encoder=False,
                                                          eval_metric='logloss', n_jobs=-1)))

if LIGHTGBM_AVAILABLE:
    base_estimators.append(('lightgbm', lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1,
                                                            max_depth=5, random_state=42,
                                                            n_jobs=-1, verbose=-1)))

ensemble_results = []

# 1. Voting (Soft)
print("\\nTraining Voting Classifier (Soft)...", end=" ")
voting_soft = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=-1)
voting_soft.fit(X_train_scaled, y_train)
y_pred_vs = voting_soft.predict(X_test_scaled)
y_pred_proba_vs = voting_soft.predict_proba(X_test_scaled)[:, 1]

ensemble_results.append({
    'Model': 'Voting (Soft)',
    'Accuracy': accuracy_score(y_test, y_pred_vs),
    'Precision': precision_score(y_test, y_pred_vs, zero_division=0),
    'Recall': recall_score(y_test, y_pred_vs, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_vs, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba_vs),
    'Model_Object': voting_soft
})
print(f"✓ AUC: {ensemble_results[-1]['AUC-ROC']:.4f}")

# 2. Stacking (LR)
print("Training Stacking (LR meta-learner)...", end=" ")
stacking_lr = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5, n_jobs=-1
)
stacking_lr.fit(X_train_scaled, y_train)
y_pred_slr = stacking_lr.predict(X_test_scaled)
y_pred_proba_slr = stacking_lr.predict_proba(X_test_scaled)[:, 1]

ensemble_results.append({
    'Model': 'Stacking (LR)',
    'Accuracy': accuracy_score(y_test, y_pred_slr),
    'Precision': precision_score(y_test, y_pred_slr, zero_division=0),
    'Recall': recall_score(y_test, y_pred_slr, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_slr, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba_slr),
    'Model_Object': stacking_lr
})
print(f"✓ AUC: {ensemble_results[-1]['AUC-ROC']:.4f}")

# 3. Stacking (RF)
print("Training Stacking (RF meta-learner)...", end=" ")
stacking_rf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    cv=5, n_jobs=-1
)
stacking_rf.fit(X_train_scaled, y_train)
y_pred_srf = stacking_rf.predict(X_test_scaled)
y_pred_proba_srf = stacking_rf.predict_proba(X_test_scaled)[:, 1]

ensemble_results.append({
    'Model': 'Stacking (RF)',
    'Accuracy': accuracy_score(y_test, y_pred_srf),
    'Precision': precision_score(y_test, y_pred_srf, zero_division=0),
    'Recall': recall_score(y_test, y_pred_srf, zero_division=0),
    'F1-Score': f1_score(y_test, y_pred_srf, zero_division=0),
    'AUC-ROC': roc_auc_score(y_test, y_pred_proba_srf),
    'Model_Object': stacking_rf
})
print(f"✓ AUC: {ensemble_results[-1]['AUC-ROC']:.4f}")

ensemble_df = pd.DataFrame(ensemble_results).sort_values('AUC-ROC', ascending=False)

print("\\n" + "="*80)
print("ENSEMBLE MODELS RESULTS")
print("="*80)
print(ensemble_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_string(index=False))

# ============================================================================
# 6. FIND BEST MODEL OVERALL
# ============================================================================

print("\\n6. BEST MODEL COMPARISON")
print("="*80)

all_models_df = pd.concat([results_df, ensemble_df], ignore_index=True)
all_models_df = all_models_df.sort_values('AUC-ROC', ascending=False)

best_model = all_models_df.iloc[0]

print(f"\\n🏆 BEST MODEL (Hybrid Imputation):")
print(f"   Model: {best_model['Model']}")
print(f"   AUC-ROC: {best_model['AUC-ROC']:.4f} ({best_model['AUC-ROC']*100:.2f}%)")
print(f"   Accuracy: {best_model['Accuracy']*100:.2f}%")
print(f"   Precision: {best_model['Precision']*100:.2f}%")
print(f"   Recall: {best_model['Recall']*100:.2f}%")
print(f"   F1-Score: {best_model['F1-Score']:.4f}")

# ============================================================================
# 7. COMPARE WITH ORIGINAL RESULTS
# ============================================================================

print("\\n7. COMPARISON WITH ORIGINAL (KNN-ONLY) IMPUTATION")
print("="*80)

# Load original best model metadata
try:
    with open('covid_results/models/model_metadata.json', 'r') as f:
        original_metadata = json.load(f)
    
    orig_auc = original_metadata['performance_metrics']['auc_roc']
    orig_model = original_metadata['model_name']
    
    print(f"\\nOriginal (KNN-only imputation):")
    print(f"   Best Model: {orig_model}")
    print(f"   AUC-ROC: {orig_auc:.4f} ({orig_auc*100:.2f}%)")
    print(f"   Imputation: KNN k=5 only (2.20% avg mean change)")
    
    print(f"\\nHybrid (KNN+MICE imputation):")
    print(f"   Best Model: {best_model['Model']}")
    print(f"   AUC-ROC: {best_model['AUC-ROC']:.4f} ({best_model['AUC-ROC']*100:.2f}%)")
    print(f"   Imputation: Hybrid KNN+MICE (0.72% avg mean change)")
    
    improvement = (best_model['AUC-ROC'] - orig_auc) * 100
    imputation_improvement = ((2.20 - 0.72) / 2.20) * 100
    
    print(f"\\n📊 IMPROVEMENTS:")
    print(f"   Model AUC: {improvement:+.2f} percentage points")
    print(f"   Imputation Quality: {imputation_improvement:.1f}% better (67% reduction in mean change)")
    
    if improvement > 0:
        print(f"\\n✅ Hybrid imputation IMPROVED model performance!")
    elif improvement < -0.5:
        print(f"\\n⚠️  Original imputation was slightly better for this model")
    else:
        print(f"\\n✓ Performance is comparable (within 0.5 percentage points)")
        
except FileNotFoundError:
    print("\\n⚠️  Original results not found for comparison")

# ============================================================================
# 8. SAVE RESULTS
# ============================================================================

print("\\n8. SAVING RESULTS")
print("="*80)

# Save best model
model_path = os.path.join(models_dir, 'best_model_hybrid.joblib')
scaler_path = os.path.join(models_dir, 'scaler_hybrid.joblib')
metadata_path = os.path.join(models_dir, 'model_metadata_hybrid.json')

dump(best_model['Model_Object'], model_path)
dump(scaler, scaler_path)

metadata = {
    'model_name': best_model['Model'],
    'model_type': type(best_model['Model_Object']).__name__,
    'imputation_method': 'Hybrid (KNN k=5 for 5 features + MICE iter=10 for 27 features)',
    'imputation_quality': '0.72% avg mean change (67% better than KNN-only)',
    'scaler': 'StandardScaler',
    'performance_metrics': {
        'accuracy': float(best_model['Accuracy']),
        'precision': float(best_model['Precision']),
        'recall': float(best_model['Recall']),
        'f1_score': float(best_model['F1-Score']),
        'auc_roc': float(best_model['AUC-ROC'])
    },
    'training_details': {
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'n_features': int(X_train.shape[1])
    },
    'feature_names': list(X.columns),
    'target_classes': ['Negative', 'Positive'],
    'knn_features': ['UREA', 'PLT1', 'NET', 'MOT', 'BAT'],
    'mice_features': [col for col in X.columns if col not in ['UREA', 'PLT1', 'NET', 'MOT', 'BAT']]
}

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✓ Model saved: {model_path}")
print(f"✓ Scaler saved: {scaler_path}")
print(f"✓ Metadata saved: {metadata_path}")

# Save results CSV
results_csv = os.path.join(output_dir, 'all_results_hybrid.csv')
all_models_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_csv(results_csv, index=False)
print(f"✓ Results table saved: {results_csv}")

# Save comparison summary
comparison_file = os.path.join(output_dir, 'comparison_summary.txt')
with open(comparison_file, 'w') as f:
    f.write("="*80 + "\\n")
    f.write("COVID-19 MODEL COMPARISON: KNN-ONLY vs HYBRID IMPUTATION\\n")
    f.write("="*80 + "\\n\\n")
    
    f.write("IMPUTATION METHODS:\\n")
    f.write("-"*80 + "\\n")
    f.write("Original: KNN k=5 for all features (2.20% avg mean change)\\n")
    f.write("Hybrid:   KNN k=5 for 5 features + MICE iter=10 for 27 features\\n")
    f.write("          (0.72% avg mean change - 67% improvement)\\n\\n")
    
    f.write("BEST MODELS:\\n")
    f.write("-"*80 + "\\n")
    if 'orig_auc' in locals():
        f.write(f"Original: {orig_model} - AUC: {orig_auc:.4f} ({orig_auc*100:.2f}%)\\n")
    f.write(f"Hybrid:   {best_model['Model']} - AUC: {best_model['AUC-ROC']:.4f} ({best_model['AUC-ROC']*100:.2f}%)\\n\\n")
    
    if 'improvement' in locals():
        f.write("IMPROVEMENTS:\\n")
        f.write("-"*80 + "\\n")
        f.write(f"Model AUC: {improvement:+.2f} percentage points\\n")
        f.write(f"Imputation Quality: {imputation_improvement:.1f}% better\\n")

print(f"✓ Comparison summary saved: {comparison_file}")

print("\\n" + "="*80)
print("✓ HYBRID IMPUTATION ANALYSIS COMPLETE!")
print("="*80)
print(f"\\nResults saved to: {output_dir}/")
print("Compare with original results in: covid_results/")
