"""
COVID-19 Ensemble Models
Builds and evaluates ensemble methods using the best base models
Saves improved model if it outperforms the existing best model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
from joblib import dump, load

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print(" COVID-19 ENSEMBLE MODELS")
print("="*80)

# Setup directories
output_dir = 'covid_results'
models_dir = os.path.join(output_dir, 'models')
plots_dir = os.path.join(output_dir, 'plots')

for dir_path in [output_dir, models_dir, plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 1. LOAD DATA AND PREVIOUS RESULTS
# ============================================================================

print("\n1. LOADING DATA AND PREVIOUS RESULTS")
print("="*80)

# Get dataset path
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'Dataset3.csv')

if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(script_dir), 'Dataset3Covid', 'Dataset3.csv')

print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"[OK] Dataset loaded: {df.shape}")

# Load previous best model info
metadata_path = os.path.join(models_dir, 'model_metadata.json')
if os.path.exists(metadata_path):
    with open(metadata_path, 'r') as f:
        prev_metadata = json.load(f)
    print(f"\n[OK] Previous best model loaded:")
    print(f"   Model: {prev_metadata['model_name']}")
    print(f"   AUC-ROC: {prev_metadata['performance_metrics']['auc_roc']:.4f}")
    print(f"   Accuracy: {prev_metadata['performance_metrics']['accuracy']:.4f}")
    prev_best_auc = prev_metadata['performance_metrics']['auc_roc']
else:
    print("\n⚠️ No previous model found")
    prev_best_auc = 0.0

# ============================================================================
# 2. PREPROCESSING
# ============================================================================

print("\n2. DATA PREPROCESSING")
print("="*80)

# Separate features and target
X = df.drop('SARSCov', axis=1)
y = df['SARSCov']

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X)}")
print(f"Missing values: {X.isnull().sum().sum()}")

# Impute missing values
if X.isnull().sum().sum() > 0:
    print("\nImputing missing values with KNN Imputer...")
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    X_imputed = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
    print(f"[OK] Imputation complete")
else:
    X_imputed = X.copy()

# Train-test split (same random state as original for consistency)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[OK] Feature scaling complete (StandardScaler)")

# ============================================================================
# 3. DEFINE BASE MODELS FOR ENSEMBLE
# ============================================================================

print("\n3. DEFINING BASE MODELS")
print("="*80)

base_estimators = []

# 1. SVM (RBF) - Often top performer
print("Base models:")
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42, 
              class_weight='balanced', C=1.0, gamma='scale')
base_estimators.append(('svm_rbf', svm_rbf))
print("  1. SVM (RBF)")

# 2. Random Forest
rf = RandomForestClassifier(
    n_estimators=150, 
    max_depth=15, 
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42, 
    class_weight='balanced', 
    n_jobs=-1
)
base_estimators.append(('random_forest', rf))
print("  2. Random Forest")

# 3. Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=150, 
    learning_rate=0.1, 
    max_depth=5,
    min_samples_split=5,
    subsample=0.8,
    random_state=42
)
base_estimators.append(('gradient_boost', gb))
print("  3. Gradient Boosting")

# 4. XGBoost
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.1, 
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        eval_metric='logloss',
        n_jobs=-1
    )
    base_estimators.append(('xgboost', xgb))
    print("  4. XGBoost")

# 5. LightGBM
if LIGHTGBM_AVAILABLE:
    lgbm = LGBMClassifier(
        n_estimators=150, 
        learning_rate=0.1, 
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42, 
        verbose=-1,
        n_jobs=-1
    )
    base_estimators.append(('lightgbm', lgbm))
    print("  5. LightGBM")

print(f"\n[OK] Total base models: {len(base_estimators)}")

# ============================================================================
# 4. TRAIN INDIVIDUAL BASE MODELS FOR COMPARISON
# ============================================================================

print("\n4. TRAINING BASE MODELS (FOR COMPARISON)")
print("="*80)

base_results = []

for name, model in base_estimators:
    print(f"\nTraining {name}...", end=" ")
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    base_results.append({
        'Model': name.replace('_', ' ').title(),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    })
    
    print(f"[OK] AUC: {auc:.4f}, Acc: {accuracy:.4f}")

base_df = pd.DataFrame(base_results).sort_values('AUC-ROC', ascending=False)
print("\nBase Models Performance:")
print(base_df.to_string(index=False))

# ============================================================================
# 5. TRAIN ENSEMBLE MODELS
# ============================================================================

print("\n5. TRAINING ENSEMBLE MODELS")
print("="*80)

ensemble_results = []

# ============================================================================
# Ensemble 1: Voting Classifier (Hard Voting)
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 1: Hard Voting Classifier")
print("-"*80)

voting_hard = VotingClassifier(
    estimators=base_estimators,
    voting='hard',
    n_jobs=-1
)

print("Training...", end=" ")
voting_hard.fit(X_train_scaled, y_train)
y_pred_vh = voting_hard.predict(X_test_scaled)

acc_vh = accuracy_score(y_test, y_pred_vh)
prec_vh = precision_score(y_test, y_pred_vh, zero_division=0)
rec_vh = recall_score(y_test, y_pred_vh, zero_division=0)
f1_vh = f1_score(y_test, y_pred_vh, zero_division=0)

print(f"[OK] Acc: {acc_vh:.4f}, F1: {f1_vh:.4f}")

ensemble_results.append({
    'Model': 'Voting (Hard)',
    'Type': 'Voting',
    'Accuracy': acc_vh,
    'Precision': prec_vh,
    'Recall': rec_vh,
    'F1-Score': f1_vh,
    'AUC-ROC': None,
    'Model_Object': voting_hard
})

# ============================================================================
# Ensemble 2: Voting Classifier (Soft Voting)
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 2: Soft Voting Classifier")
print("-"*80)

voting_soft = VotingClassifier(
    estimators=base_estimators,
    voting='soft',
    n_jobs=-1
)

print("Training...", end=" ")
voting_soft.fit(X_train_scaled, y_train)
y_pred_vs = voting_soft.predict(X_test_scaled)
y_pred_proba_vs = voting_soft.predict_proba(X_test_scaled)[:, 1]

acc_vs = accuracy_score(y_test, y_pred_vs)
prec_vs = precision_score(y_test, y_pred_vs, zero_division=0)
rec_vs = recall_score(y_test, y_pred_vs, zero_division=0)
f1_vs = f1_score(y_test, y_pred_vs, zero_division=0)
auc_vs = roc_auc_score(y_test, y_pred_proba_vs)

print(f"[OK] Acc: {acc_vs:.4f}, AUC: {auc_vs:.4f}")

ensemble_results.append({
    'Model': 'Voting (Soft)',
    'Type': 'Voting',
    'Accuracy': acc_vs,
    'Precision': prec_vs,
    'Recall': rec_vs,
    'F1-Score': f1_vs,
    'AUC-ROC': auc_vs,
    'Model_Object': voting_soft
})

# ============================================================================
# Ensemble 3: Stacking with Logistic Regression Meta-Learner
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 3: Stacking Classifier (Meta: Logistic Regression)")
print("-"*80)

stacking_lr = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

print("Training (with 5-fold CV)...", end=" ")
stacking_lr.fit(X_train_scaled, y_train)
y_pred_slr = stacking_lr.predict(X_test_scaled)
y_pred_proba_slr = stacking_lr.predict_proba(X_test_scaled)[:, 1]

acc_slr = accuracy_score(y_test, y_pred_slr)
prec_slr = precision_score(y_test, y_pred_slr, zero_division=0)
rec_slr = recall_score(y_test, y_pred_slr, zero_division=0)
f1_slr = f1_score(y_test, y_pred_slr, zero_division=0)
auc_slr = roc_auc_score(y_test, y_pred_proba_slr)

print(f"[OK] Acc: {acc_slr:.4f}, AUC: {auc_slr:.4f}")

ensemble_results.append({
    'Model': 'Stacking (LR)',
    'Type': 'Stacking',
    'Accuracy': acc_slr,
    'Precision': prec_slr,
    'Recall': rec_slr,
    'F1-Score': f1_slr,
    'AUC-ROC': auc_slr,
    'Model_Object': stacking_lr
})

# ============================================================================
# Ensemble 4: Stacking with Random Forest Meta-Learner
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 4: Stacking Classifier (Meta: Random Forest)")
print("-"*80)

stacking_rf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=RandomForestClassifier(
        n_estimators=50, 
        max_depth=10,
        random_state=42, 
        n_jobs=-1
    ),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

print("Training (with 5-fold CV)...", end=" ")
stacking_rf.fit(X_train_scaled, y_train)
y_pred_srf = stacking_rf.predict(X_test_scaled)
y_pred_proba_srf = stacking_rf.predict_proba(X_test_scaled)[:, 1]

acc_srf = accuracy_score(y_test, y_pred_srf)
prec_srf = precision_score(y_test, y_pred_srf, zero_division=0)
rec_srf = recall_score(y_test, y_pred_srf, zero_division=0)
f1_srf = f1_score(y_test, y_pred_srf, zero_division=0)
auc_srf = roc_auc_score(y_test, y_pred_proba_srf)

print(f"[OK] Acc: {acc_srf:.4f}, AUC: {auc_srf:.4f}")

ensemble_results.append({
    'Model': 'Stacking (RF)',
    'Type': 'Stacking',
    'Accuracy': acc_srf,
    'Precision': prec_srf,
    'Recall': rec_srf,
    'F1-Score': f1_srf,
    'AUC-ROC': auc_srf,
    'Model_Object': stacking_rf
})

# ============================================================================
# Ensemble 5: Stacking with Gradient Boosting Meta-Learner
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 5: Stacking Classifier (Meta: Gradient Boosting)")
print("-"*80)

stacking_gb = StackingClassifier(
    estimators=base_estimators,
    final_estimator=GradientBoostingClassifier(
        n_estimators=50, 
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    cv=5,
    n_jobs=-1,
    passthrough=False
)

print("Training (with 5-fold CV)...", end=" ")
stacking_gb.fit(X_train_scaled, y_train)
y_pred_sgb = stacking_gb.predict(X_test_scaled)
y_pred_proba_sgb = stacking_gb.predict_proba(X_test_scaled)[:, 1]

acc_sgb = accuracy_score(y_test, y_pred_sgb)
prec_sgb = precision_score(y_test, y_pred_sgb, zero_division=0)
rec_sgb = recall_score(y_test, y_pred_sgb, zero_division=0)
f1_sgb = f1_score(y_test, y_pred_sgb, zero_division=0)
auc_sgb = roc_auc_score(y_test, y_pred_proba_sgb)

print(f"[OK] Acc: {acc_sgb:.4f}, AUC: {auc_sgb:.4f}")

ensemble_results.append({
    'Model': 'Stacking (GB)',
    'Type': 'Stacking',
    'Accuracy': acc_sgb,
    'Precision': prec_sgb,
    'Recall': rec_sgb,
    'F1-Score': f1_sgb,
    'AUC-ROC': auc_sgb,
    'Model_Object': stacking_gb
})

# ============================================================================
# 6. RESULTS COMPARISON
# ============================================================================

print("\n6. RESULTS COMPARISON")
print("="*80)

ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df = ensemble_df.sort_values('AUC-ROC', ascending=False)

print("\nEnsemble Models Performance:")
display_cols = ['Model', 'Type', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
print(ensemble_df[display_cols].to_string(index=False))

# Find best ensemble
best_ensemble_idx = ensemble_df['AUC-ROC'].idxmax()
best_ensemble = ensemble_df.loc[best_ensemble_idx]

print(f"\n{'='*80}")
print("BEST ENSEMBLE MODEL")
print(f"{'='*80}")
print(f"Model: {best_ensemble['Model']}")
print(f"Type: {best_ensemble['Type']}")
print(f"Accuracy: {best_ensemble['Accuracy']*100:.2f}%")
print(f"Precision: {best_ensemble['Precision']*100:.2f}%")
print(f"Recall: {best_ensemble['Recall']*100:.2f}%")
print(f"F1-Score: {best_ensemble['F1-Score']:.4f}")
print(f"AUC-ROC: {best_ensemble['AUC-ROC']:.4f}")

# ============================================================================
# 7. COMPARE WITH PREVIOUS BEST MODEL
# ============================================================================

print(f"\n{'='*80}")
print("COMPARISON WITH PREVIOUS BEST MODEL")
print(f"{'='*80}")

print(f"\nPrevious best AUC: {prev_best_auc:.4f}")
print(f"Best ensemble AUC: {best_ensemble['AUC-ROC']:.4f}")

improvement = (best_ensemble['AUC-ROC'] - prev_best_auc) * 100
print(f"Improvement: {improvement:+.2f} percentage points")

save_new_model = False

if best_ensemble['AUC-ROC'] > prev_best_auc:
    print("\n🎉 NEW BEST MODEL FOUND!")
    print(f"The {best_ensemble['Model']} ensemble outperforms the previous best model!")
    save_new_model = True
else:
    print(f"\n[OK] Previous model is still better by {-improvement:.2f} percentage points")
    print("Ensemble models will still be saved for reference.")

# ============================================================================
# 8. DETAILED EVALUATION OF BEST ENSEMBLE
# ============================================================================

print("\n8. DETAILED EVALUATION OF BEST ENSEMBLE")
print("="*80)

best_model = best_ensemble['Model_Object']

# Get predictions based on model type
if best_ensemble['Model'] == 'Voting (Hard)':
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_proba_best = None
else:
    y_pred_best = best_model.predict(X_test_scaled)
    y_pred_proba_best = best_model.predict_proba(X_test_scaled)[:, 1]

# Classification report
print(f"\nClassification Report ({best_ensemble['Model']}):")
print(classification_report(y_test, y_pred_best, target_names=['Negative', 'Positive']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
print(f"\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives: {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives: {tp}")

specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f"\nSpecificity: {specificity:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================

print("\n9. GENERATING VISUALIZATIONS")
print("="*80)

# 9.1 Ensemble comparison bar plot
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(ensemble_df))
width = 0.15

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    values = ensemble_df[metric].values
    ax.bar(x + i*width, values, width, label=metric, color=color, alpha=0.8)

ax.set_xlabel('Ensemble Models', fontweight='bold', fontsize=12)
ax.set_ylabel('Score', fontweight='bold', fontsize=12)
ax.set_title('Ensemble Models Performance Comparison', fontweight='bold', fontsize=14)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(ensemble_df['Model'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ensemble_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9.2 Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            linewidths=2, linecolor='black')
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {best_ensemble["Model"]}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ensemble_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 9.3 ROC Curve (if probabilities available)
if y_pred_proba_best is not None:
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC curve (AUC = {best_ensemble["AUC-ROC"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    plt.title(f'ROC Curve - {best_ensemble["Model"]}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'ensemble_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 9.4 AUC Score comparison
plt.figure(figsize=(10, 6))
models_with_auc = ensemble_df[ensemble_df['AUC-ROC'].notna()]
colors = plt.cm.viridis(np.linspace(0, 1, len(models_with_auc)))
bars = plt.bar(models_with_auc['Model'], models_with_auc['AUC-ROC'], 
               color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('AUC-ROC Score', fontsize=12, fontweight='bold')
plt.title('Ensemble Models AUC-ROC Scores', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim([0.85, max(models_with_auc['AUC-ROC']) + 0.02])
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, models_with_auc['AUC-ROC']):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'ensemble_auc_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"[OK] Visualizations saved to: {plots_dir}/")

# ============================================================================
# 10. SAVE MODEL
# ============================================================================

print("\n10. SAVING MODELS")
print("="*80)

if save_new_model:
    # Save as best model (replace previous)
    model_path = os.path.join(models_dir, 'best_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    
    dump(best_model, model_path)
    dump(scaler, scaler_path)
    
    metadata = {
        'model_name': best_ensemble['Model'],
        'model_type': type(best_model).__name__,
        'scaler': 'StandardScaler',
        'ensemble_type': best_ensemble['Type'],
        'base_models': [name for name, _ in base_estimators],
        'performance_metrics': {
            'accuracy': float(best_ensemble['Accuracy']),
            'precision': float(best_ensemble['Precision']),
            'recall': float(best_ensemble['Recall']),
            'f1_score': float(best_ensemble['F1-Score']),
            'auc_roc': float(best_ensemble['AUC-ROC'])
        },
        'training_details': {
            'n_samples_train': int(len(X_train)),
            'n_samples_test': int(len(X_test)),
            'n_features': int(X_train.shape[1]),
            'imputation_method': 'KNNImputer (k=5)'
        },
        'feature_names': list(X.columns),
        'target_classes': ['Negative', 'Positive']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] NEW BEST MODEL SAVED!")
    print(f"   Model: {model_path}")
    print(f"   Scaler: {scaler_path}")
    print(f"   Metadata: {metadata_path}")
else:
    # Save ensemble models separately
    ensemble_model_path = os.path.join(models_dir, f'ensemble_{best_ensemble["Model"].lower().replace(" ", "_")}.joblib')
    ensemble_metadata_path = os.path.join(models_dir, 'ensemble_models_metadata.json')
    
    dump(best_model, ensemble_model_path)
    
    ensemble_metadata = {
        'best_ensemble': best_ensemble['Model'],
        'all_ensembles': []
    }
    
    for _, row in ensemble_df.iterrows():
        ensemble_metadata['all_ensembles'].append({
            'model_name': row['Model'],
            'type': row['Type'],
            'metrics': {
                'accuracy': float(row['Accuracy']),
                'precision': float(row['Precision']),
                'recall': float(row['Recall']),
                'f1_score': float(row['F1-Score']),
                'auc_roc': float(row['AUC-ROC']) if row['AUC-ROC'] is not None else None
            }
        })
    
    with open(ensemble_metadata_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)
    
    print(f"[OK] Ensemble models saved for reference")
    print(f"   Best ensemble: {ensemble_model_path}")
    print(f"   Metadata: {ensemble_metadata_path}")

# Save results table
results_csv_path = os.path.join(output_dir, 'ensemble_results.csv')
ensemble_df[display_cols].to_csv(results_csv_path, index=False)
print(f"[OK] Results table: {results_csv_path}")

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "="*80)
print(" ENSEMBLE ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print(f"\n🤖 Ensemble Methods Tested:")
print(f"   - Voting (Hard): Majority vote")
print(f"   - Voting (Soft): Weighted probability average")
print(f"   - Stacking (LR): Logistic Regression meta-learner")
print(f"   - Stacking (RF): Random Forest meta-learner")
print(f"   - Stacking (GB): Gradient Boosting meta-learner")

print(f"\n🏆 Best Ensemble:")
print(f"   Model: {best_ensemble['Model']}")
print(f"   AUC-ROC: {best_ensemble['AUC-ROC']:.4f} ({best_ensemble['AUC-ROC']*100:.2f}%)")
print(f"   Accuracy: {best_ensemble['Accuracy']*100:.2f}%")
print(f"   F1-Score: {best_ensemble['F1-Score']:.4f}")

if save_new_model:
    print(f"\n✅ NEW BEST MODEL SAVED!")
    print(f"   Improvement: +{improvement:.2f} percentage points")
else:
    print(f"\n[OK] Previous model remains best")
    print(f"   (Ensemble still saved for reference)")

print(f"\n📁 Output Files:")
print(f"   - Models: {models_dir}/")
print(f"   - Plots: {plots_dir}/")
print(f"   - Results: {results_csv_path}")

print("\n" + "="*80)
print("[OK] Ensemble Analysis Complete!")
print("="*80)
