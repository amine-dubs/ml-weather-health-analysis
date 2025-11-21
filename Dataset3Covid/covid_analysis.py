"""
COVID-19 Classification Analysis
Dataset: Dataset3.csv
Objective: Predict SARS-CoV-2 infection from clinical laboratory tests
Methods: EDA, Preprocessing, Multiple Classification Models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
from joblib import dump

# Machine Learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)

# Linear Models
from sklearn.linear_model import LogisticRegression

# Non-linear Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.naive_bayes import GaussianNB

# Advanced models
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print(" COVID-19 SARS-CoV-2 CLASSIFICATION ANALYSIS")
print("="*80)

# Create output directories - NEW FOLDER FOR HYBRID IMPUTATION
output_dir = 'covid_results_hybrid'
models_dir = os.path.join(output_dir, 'models')
plots_dir = os.path.join(output_dir, 'plots')

for dir_path in [output_dir, models_dir, plots_dir]:
    os.makedirs(dir_path, exist_ok=True)

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================

print("\n1. LOADING AND INITIAL EXPLORATION")
print("="*80)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'Dataset3.csv')

# Check if file exists, if not try parent directory
if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(script_dir), 'Dataset3Covid', 'Dataset3.csv')

print(f"Loading dataset from: {dataset_path}")
df = pd.read_csv(dataset_path)
print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {df.shape[0]}")
print(f"Number of features: {df.shape[1] - 1}")  # Excluding target

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\nTarget variable (SARSCov) distribution:")
target_counts = df['SARSCov'].value_counts()
print(target_counts)
print("\nTarget percentages:")
print(df['SARSCov'].value_counts(normalize=True) * 100)

# Check for class imbalance
imbalance_ratio = target_counts.max() / target_counts.min()
print(f"\nClass imbalance ratio: {imbalance_ratio:.2f}")
if imbalance_ratio > 1.5:
    print("⚠️ Dataset is imbalanced - will handle during modeling")

# ============================================================================
# 2. MISSING VALUES ANALYSIS
# ============================================================================

print("\n2. MISSING VALUES ANALYSIS")
print("="*80)

missing_counts = df.isnull().sum()
missing_percentage = (missing_counts / len(df)) * 100

missing_df = pd.DataFrame({
    'Feature': missing_counts.index,
    'Missing_Count': missing_counts.values,
    'Percentage': missing_percentage.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

print(f"\nTotal missing values: {df.isnull().sum().sum()}")
print(f"Complete cases: {(~df.isnull().any(axis=1)).sum()} ({(~df.isnull().any(axis=1)).sum()/len(df)*100:.2f}%)")

print("\nFeatures with missing values:")
print(missing_df.to_string(index=False))

# Visualize missing values
plt.figure(figsize=(14, 8))
missing_plot_df = missing_df.head(20)  # Top 20 features with most missing
plt.barh(missing_plot_df['Feature'], missing_plot_df['Percentage'], color='coral')
plt.xlabel('Percentage Missing (%)', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Missing Values by Feature (Top 20)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, (feat, pct) in enumerate(zip(missing_plot_df['Feature'], missing_plot_df['Percentage'])):
    plt.text(pct + 0.5, i, f'{pct:.1f}%', va='center')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'missing_values.png'), dpi=300, bbox_inches='tight')
plt.close()

# ============================================================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

print("\n3. EXPLORATORY DATA ANALYSIS")
print("="*80)

# Statistical description
print("\nStatistical Summary (numerical features):")
print(df.describe().T)

# Separate features and target
X = df.drop('SARSCov', axis=1)
y = df['SARSCov']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")

# 3.1 Distribution of numerical features
print("\n3.1 Feature Distributions")

numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
print(f"Number of numerical features: {len(numerical_features)}")

# Plot distributions for key features
key_features = ['Age', 'WBC', 'RBC', 'HGB', 'PLT1', 'CREA', 'GLU', 'PCR']
available_features = [f for f in key_features if f in numerical_features]

if len(available_features) >= 6:
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(available_features[:9]):
        axes[i].hist(X[feature].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Distribution of {feature}', fontweight='bold')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(available_features), 9):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 3.2 Target distribution visualization
plt.figure(figsize=(10, 6))
target_labels = ['Negative', 'Positive']
colors = ['#2ecc71', '#e74c3c']
plt.bar(target_labels, target_counts.values, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.title('SARS-CoV-2 Test Results Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, (label, count) in enumerate(zip(target_labels, target_counts.values)):
    plt.text(i, count + 20, f'{count}\n({count/len(df)*100:.1f}%)', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'target_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3.3 Feature correlation with target
print("\n3.3 Feature Correlation with Target")

# Calculate correlation for complete cases
correlations = []
for col in numerical_features:
    if col in X.columns:
        # Remove NaN values for correlation calculation
        valid_mask = X[col].notna() & y.notna()
        if valid_mask.sum() > 0:
            corr = np.corrcoef(X.loc[valid_mask, col], y[valid_mask])[0, 1]
            correlations.append({'Feature': col, 'Correlation': corr})

corr_df = pd.DataFrame(correlations).sort_values('Correlation', key=abs, ascending=False)
print("\nTop 15 features correlated with COVID-19 status:")
print(corr_df.head(15).to_string(index=False))

# Plot top correlations
plt.figure(figsize=(12, 8))
top_corr = corr_df.head(20)
colors = ['red' if x < 0 else 'green' for x in top_corr['Correlation']]
plt.barh(top_corr['Feature'], top_corr['Correlation'], color=colors, alpha=0.7)
plt.xlabel('Correlation with COVID-19 Positive', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 20 Features Correlated with SARS-CoV-2 Status', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3.4 Box plots for key features by target
print("\n3.4 Feature Distributions by COVID-19 Status")

key_plot_features = [f for f in ['Age', 'WBC', 'LY', 'PCR', 'CREA', 'LDH'] if f in X.columns]

if len(key_plot_features) >= 4:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(key_plot_features[:6]):
        data_to_plot = [X[y==0][feature].dropna(), X[y==1][feature].dropna()]
        bp = axes[i].boxplot(data_to_plot, labels=['Negative', 'Positive'], 
                             patch_artist=True, showmeans=True)
        for patch, color in zip(bp['boxes'], ['#2ecc71', '#e74c3c']):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[i].set_ylabel(feature, fontweight='bold')
        axes[i].set_title(f'{feature} by COVID-19 Status', fontweight='bold')
        axes[i].grid(alpha=0.3)
    
    for i in range(len(key_plot_features), 6):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'boxplots_by_target.png'), dpi=300, bbox_inches='tight')
    plt.close()

# 3.5 Correlation heatmap
print("\n3.5 Feature Correlation Matrix")

# Select subset of features for heatmap (too many features make it unreadable)
heatmap_features = corr_df.head(15)['Feature'].tolist()

plt.figure(figsize=(12, 10))
correlation_matrix = X[heatmap_features].corr()
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
            cmap='coolwarm', center=0, square=True, 
            linewidths=1, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Top Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ EDA plots saved to: {plots_dir}/")

# ============================================================================
# 4. PREPROCESSING
# ============================================================================

print("\n4. DATA PREPROCESSING")
print("="*80)

# 4.1 Handle missing values with HYBRID Imputation (KNN + MICE)
print("\n4.1 Imputing Missing Values with HYBRID Strategy (KNN + MICE)")
print("    Based on optimization: 0.72% average mean change")

print(f"Missing values before imputation: {X.isnull().sum().sum()}")

# Hybrid imputation configuration (from hybrid_imputation.py results)
# KNN features: UREA, PLT1, NET, MOT, BAT
# MICE features: All others
knn_features = ['UREA', 'PLT1', 'NET', 'MOT', 'BAT']
mice_features = [col for col in X.columns if col not in knn_features]

print(f"  - KNN imputation: {len(knn_features)} features")
print(f"  - MICE imputation: {len(mice_features)} features")

# Apply KNN imputation
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_knn = pd.DataFrame(
    knn_imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Apply MICE imputation
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
X_mice = pd.DataFrame(
    mice_imputer.fit_transform(X),
    columns=X.columns,
    index=X.index
)

# Combine: Use KNN for KNN features, MICE for MICE features
X_imputed = X.copy()
for col in knn_features:
    if col in X.columns:
        X_imputed[col] = X_knn[col]
for col in mice_features:
    X_imputed[col] = X_mice[col]

print(f"Missing values after imputation: {X_imputed.isnull().sum().sum()}")
print("✓ Hybrid imputation completed (0.72% avg mean change)")

# 4.2 Feature scaling comparison
print("\n4.2 Feature Scaling")

print("Will compare three scaling methods:")
print("  1. StandardScaler (Z-score normalization)")
print("  2. MinMaxScaler (0-1 normalization)")
print("  3. RobustScaler (robust to outliers)")

# ============================================================================
# 5. TRAIN-TEST SPLIT
# ============================================================================

print("\n5. TRAIN-TEST SPLIT")
print("="*80)

# Split data (80-20 split, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nTraining set class distribution:")
print(pd.Series(y_train).value_counts())
print(f"\nTest set class distribution:")
print(pd.Series(y_test).value_counts())

# ============================================================================
# 6. MODEL TRAINING AND EVALUATION
# ============================================================================

print("\n6. MODEL TRAINING AND EVALUATION")
print("="*80)

# Initialize scalers
scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler(),
    'NoScaling': None
}

# Initialize models
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

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    models['XGBoost'] = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                     random_state=42, eval_metric='logloss', n_jobs=-1)
    print("✓ XGBoost added")

# Add LightGBM if available
if LIGHTGBM_AVAILABLE:
    models['LightGBM'] = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                       random_state=42, verbose=-1, n_jobs=-1)
    print("✓ LightGBM added")

print(f"\nTotal models to train: {len(models)}")
print(f"Scaling methods: {len(scalers)}")
print(f"Total configurations: {len(models) * len(scalers)}")

# Store all results
all_results = []

print("\nStarting model training...")
print("="*80)

for scaler_name, scaler in scalers.items():
    print(f"\n{'='*80}")
    print(f"SCALING METHOD: {scaler_name}")
    print(f"{'='*80}")
    
    # Apply scaling
    if scaler is not None:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
    
    for model_name, model in models.items():
        print(f"\nTraining: {model_name} with {scaler_name}...", end=" ")
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if y_pred_proba is not None:
                auc = roc_auc_score(y_test, y_pred_proba)
            else:
                auc = None
            
            # Store results
            result = {
                'Model': model_name,
                'Scaler': scaler_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'AUC-ROC': auc,
                'Model_Object': model,
                'Scaler_Object': scaler
            }
            all_results.append(result)
            
            auc_str = f"{auc:.4f}" if auc is not None else "N/A"
            print(f"✓ Acc: {accuracy:.4f}, AUC: {auc_str}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE")
print("="*80)

# ============================================================================
# 7. RESULTS ANALYSIS
# ============================================================================

print("\n7. RESULTS ANALYSIS")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('AUC-ROC', ascending=False)

print("\nTop 20 Model Configurations (by AUC-ROC):")
display_cols = ['Model', 'Scaler', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
print(results_df[display_cols].head(20).to_string(index=False))

# Find best model overall
best_idx = results_df['AUC-ROC'].idxmax()
best_result = results_df.loc[best_idx]

print(f"\n{'='*80}")
print("BEST MODEL")
print(f"{'='*80}")
print(f"Model: {best_result['Model']}")
print(f"Scaler: {best_result['Scaler']}")
print(f"Accuracy: {best_result['Accuracy']:.4f}")
print(f"Precision: {best_result['Precision']:.4f}")
print(f"Recall: {best_result['Recall']:.4f}")
print(f"F1-Score: {best_result['F1-Score']:.4f}")
print(f"AUC-ROC: {best_result['AUC-ROC']:.4f}")

# Get best model predictions for detailed analysis
best_model = best_result['Model_Object']
best_scaler = best_result['Scaler_Object']

if best_scaler is not None:
    X_test_best = best_scaler.transform(X_test)
else:
    X_test_best = X_test.values

y_pred_best = best_model.predict(X_test_best)
y_pred_proba_best = best_model.predict_proba(X_test_best)[:, 1]

# Detailed classification report
print(f"\nDetailed Classification Report (Best Model):")
print(classification_report(y_test, y_pred_best, target_names=['Negative', 'Positive']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print(f"\nConfusion Matrix:")
print(cm)

# ============================================================================
# 7.5 ENSEMBLE METHODS
# ============================================================================

print("\n" + "="*80)
print("7.5 TRAINING ENSEMBLE MODELS")
print("="*80)

# Get top 3 models for ensembling (excluding ensemble models if any)
top_models_data = results_df.head(10)
print("\nBuilding ensembles from top performing models...")

# We'll use StandardScaler for ensembles (best performing overall)
scaler_ensemble = StandardScaler()
X_train_ensemble = scaler_ensemble.fit_transform(X_train)
X_test_ensemble = scaler_ensemble.transform(X_test)

# Define base models for ensemble
base_estimators = []

# Add best performing models
print("\nBase models for ensemble:")

# 1. SVM (RBF) - typically top performer
svm_rbf = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
base_estimators.append(('svm_rbf', svm_rbf))
print("  1. SVM (RBF)")

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, 
                            class_weight='balanced', n_jobs=-1)
base_estimators.append(('random_forest', rf))
print("  2. Random Forest")

# 3. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, 
                                max_depth=5, random_state=42)
base_estimators.append(('gradient_boost', gb))
print("  3. Gradient Boosting")

# 4. XGBoost (if available)
if XGBOOST_AVAILABLE:
    xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                       random_state=42, eval_metric='logloss', n_jobs=-1)
    base_estimators.append(('xgboost', xgb))
    print("  4. XGBoost")

# 5. LightGBM (if available)
if LIGHTGBM_AVAILABLE:
    lgbm = LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                         random_state=42, verbose=-1, n_jobs=-1)
    base_estimators.append(('lightgbm', lgbm))
    print("  5. LightGBM")

print(f"\nTotal base models: {len(base_estimators)}")

# ============================================================================
# Ensemble 1: Voting Classifier (Hard Voting)
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 1: Voting Classifier (Hard Voting)")
print("-"*80)

voting_hard = VotingClassifier(
    estimators=base_estimators,
    voting='hard',
    n_jobs=-1
)

print("Training Hard Voting Classifier...", end=" ")
voting_hard.fit(X_train_ensemble, y_train)
y_pred_voting_hard = voting_hard.predict(X_test_ensemble)
y_pred_proba_voting_hard = None  # Hard voting doesn't have probabilities

acc_vh = accuracy_score(y_test, y_pred_voting_hard)
prec_vh = precision_score(y_test, y_pred_voting_hard, zero_division=0)
rec_vh = recall_score(y_test, y_pred_voting_hard, zero_division=0)
f1_vh = f1_score(y_test, y_pred_voting_hard, zero_division=0)

print(f"✓ Acc: {acc_vh:.4f}, F1: {f1_vh:.4f}")

# ============================================================================
# Ensemble 2: Voting Classifier (Soft Voting)
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 2: Voting Classifier (Soft Voting)")
print("-"*80)

voting_soft = VotingClassifier(
    estimators=base_estimators,
    voting='soft',
    n_jobs=-1
)

print("Training Soft Voting Classifier...", end=" ")
voting_soft.fit(X_train_ensemble, y_train)
y_pred_voting_soft = voting_soft.predict(X_test_ensemble)
y_pred_proba_voting_soft = voting_soft.predict_proba(X_test_ensemble)[:, 1]

acc_vs = accuracy_score(y_test, y_pred_voting_soft)
prec_vs = precision_score(y_test, y_pred_voting_soft, zero_division=0)
rec_vs = recall_score(y_test, y_pred_voting_soft, zero_division=0)
f1_vs = f1_score(y_test, y_pred_voting_soft, zero_division=0)
auc_vs = roc_auc_score(y_test, y_pred_proba_voting_soft)

print(f"✓ Acc: {acc_vs:.4f}, AUC: {auc_vs:.4f}")

# ============================================================================
# Ensemble 3: Stacking Classifier with Logistic Regression
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 3: Stacking Classifier (Meta: Logistic Regression)")
print("-"*80)

stacking_lr = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)

print("Training Stacking Classifier (LR meta-learner)...", end=" ")
stacking_lr.fit(X_train_ensemble, y_train)
y_pred_stacking_lr = stacking_lr.predict(X_test_ensemble)
y_pred_proba_stacking_lr = stacking_lr.predict_proba(X_test_ensemble)[:, 1]

acc_sl = accuracy_score(y_test, y_pred_stacking_lr)
prec_sl = precision_score(y_test, y_pred_stacking_lr, zero_division=0)
rec_sl = recall_score(y_test, y_pred_stacking_lr, zero_division=0)
f1_sl = f1_score(y_test, y_pred_stacking_lr, zero_division=0)
auc_sl = roc_auc_score(y_test, y_pred_proba_stacking_lr)

print(f"✓ Acc: {acc_sl:.4f}, AUC: {auc_sl:.4f}")

# ============================================================================
# Ensemble 4: Stacking Classifier with Random Forest
# ============================================================================

print("\n" + "-"*80)
print("Ensemble 4: Stacking Classifier (Meta: Random Forest)")
print("-"*80)

stacking_rf = StackingClassifier(
    estimators=base_estimators,
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    cv=5,
    n_jobs=-1
)

print("Training Stacking Classifier (RF meta-learner)...", end=" ")
stacking_rf.fit(X_train_ensemble, y_train)
y_pred_stacking_rf = stacking_rf.predict(X_test_ensemble)
y_pred_proba_stacking_rf = stacking_rf.predict_proba(X_test_ensemble)[:, 1]

acc_srf = accuracy_score(y_test, y_pred_stacking_rf)
prec_srf = precision_score(y_test, y_pred_stacking_rf, zero_division=0)
rec_srf = recall_score(y_test, y_pred_stacking_rf, zero_division=0)
f1_srf = f1_score(y_test, y_pred_stacking_rf, zero_division=0)
auc_srf = roc_auc_score(y_test, y_pred_proba_stacking_rf)

print(f"✓ Acc: {acc_srf:.4f}, AUC: {auc_srf:.4f}")

# ============================================================================
# Add ensemble results to main results
# ============================================================================

print("\n" + "="*80)
print("ENSEMBLE RESULTS SUMMARY")
print("="*80)

ensemble_results = [
    {
        'Model': 'Voting (Hard)',
        'Scaler': 'StandardScaler',
        'Accuracy': acc_vh,
        'Precision': prec_vh,
        'Recall': rec_vh,
        'F1-Score': f1_vh,
        'AUC-ROC': None,
        'Model_Object': voting_hard,
        'Scaler_Object': scaler_ensemble
    },
    {
        'Model': 'Voting (Soft)',
        'Scaler': 'StandardScaler',
        'Accuracy': acc_vs,
        'Precision': prec_vs,
        'Recall': rec_vs,
        'F1-Score': f1_vs,
        'AUC-ROC': auc_vs,
        'Model_Object': voting_soft,
        'Scaler_Object': scaler_ensemble
    },
    {
        'Model': 'Stacking (LR)',
        'Scaler': 'StandardScaler',
        'Accuracy': acc_sl,
        'Precision': prec_sl,
        'Recall': rec_sl,
        'F1-Score': f1_sl,
        'AUC-ROC': auc_sl,
        'Model_Object': stacking_lr,
        'Scaler_Object': scaler_ensemble
    },
    {
        'Model': 'Stacking (RF)',
        'Scaler': 'StandardScaler',
        'Accuracy': acc_srf,
        'Precision': prec_srf,
        'Recall': rec_srf,
        'F1-Score': f1_srf,
        'AUC-ROC': auc_srf,
        'Model_Object': stacking_rf,
        'Scaler_Object': scaler_ensemble
    }
]

ensemble_df = pd.DataFrame(ensemble_results)
print("\nEnsemble Model Performance:")
print(ensemble_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']].to_string(index=False))

# Combine all results
all_results.extend(ensemble_results)
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('AUC-ROC', ascending=False)

print("\n" + "="*80)
print("TOP 10 MODELS (Including Ensembles)")
print("="*80)
print(results_df[display_cols].head(10).to_string(index=False))

# Update best model if ensemble is better
best_idx = results_df['AUC-ROC'].idxmax()
best_result_final = results_df.loc[best_idx]

if best_result_final['Model'] != best_result['Model']:
    print("\n🎉 NEW BEST MODEL FOUND!")
    print(f"Previous best: {best_result['Model']} (AUC: {best_result['AUC-ROC']:.4f})")
    print(f"New best: {best_result_final['Model']} (AUC: {best_result_final['AUC-ROC']:.4f})")
    print(f"Improvement: +{(best_result_final['AUC-ROC'] - best_result['AUC-ROC'])*100:.2f} percentage points")
    
    # Update best model references
    best_result = best_result_final
    best_model = best_result['Model_Object']
    best_scaler = best_result['Scaler_Object']
    
    if best_scaler is not None:
        X_test_best = best_scaler.transform(X_test)
    else:
        X_test_best = X_test.values
    
    y_pred_best = best_model.predict(X_test_best)
    y_pred_proba_best = best_model.predict_proba(X_test_best)[:, 1]
    
    print("\nUpdated Best Model Classification Report:")
    print(classification_report(y_test, y_pred_best, target_names=['Negative', 'Positive']))
    
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"\nUpdated Confusion Matrix:")
    print(cm)
else:
    print(f"\nBest model remains: {best_result['Model']} (AUC: {best_result['AUC-ROC']:.4f})")

# ============================================================================
# 8. VISUALIZATIONS
# ============================================================================

print("\n8. GENERATING VISUALIZATIONS")
print("="*80)

# 8.1 Model Comparison - AUC Scores
plt.figure(figsize=(14, 10))
top_20 = results_df.head(20)
model_labels = [f"{row['Model']}\n({row['Scaler']})" for _, row in top_20.iterrows()]
colors = plt.cm.viridis(np.linspace(0, 1, len(top_20)))
bars = plt.barh(range(len(top_20)), top_20['AUC-ROC'], color=colors, alpha=0.8)
plt.yticks(range(len(top_20)), model_labels, fontsize=9)
plt.xlabel('AUC-ROC Score', fontsize=12, fontweight='bold')
plt.title('Top 20 Model Configurations by AUC-ROC Score', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, top_20['AUC-ROC'])):
    plt.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'model_comparison_auc.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8.2 Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
plt.ylabel('Actual', fontsize=12, fontweight='bold')
plt.xlabel('Predicted', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {best_result["Model"]}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8.3 ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_best)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {best_result["AUC-ROC"]:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title(f'ROC Curve - {best_result["Model"]}', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8.4 Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba_best)
plt.figure(figsize=(10, 8))
plt.plot(recall_vals, precision_vals, color='blue', lw=2,
         label=f'PR curve (F1 = {best_result["F1-Score"]:.4f})')
plt.xlabel('Recall', fontsize=12, fontweight='bold')
plt.ylabel('Precision', fontsize=12, fontweight='bold')
plt.title(f'Precision-Recall Curve - {best_result["Model"]}', fontsize=14, fontweight='bold')
plt.legend(loc="lower left", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
plt.close()

# 8.5 Metrics Comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
for idx, metric in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    top_10 = results_df.nsmallest(10, metric, keep='all') if idx == 1 else results_df.nlargest(10, metric, keep='all')
    model_labels = [f"{row['Model'][:15]}\n({row['Scaler'][:8]})" for _, row in top_10.iterrows()]
    ax.barh(range(len(top_10)), top_10[metric], color=plt.cm.plasma(np.linspace(0, 1, len(top_10))))
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(model_labels, fontsize=8)
    ax.set_xlabel(metric, fontweight='bold')
    ax.set_title(f'Top 10 Models by {metric}', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Visualizations saved to: {plots_dir}/")

# ============================================================================
# 9. SAVE BEST MODEL
# ============================================================================

print("\n9. SAVING BEST MODEL")
print("="*80)

# Save model
model_path = os.path.join(models_dir, 'best_model.joblib')
dump(best_model, model_path)
print(f"✓ Model saved: {model_path}")

# Save scaler if used
if best_scaler is not None:
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    dump(best_scaler, scaler_path)
    print(f"✓ Scaler saved: {scaler_path}")

# Save metadata
metadata = {
    'model_name': best_result['Model'],
    'model_type': type(best_model).__name__,
    'scaler': best_result['Scaler'],
    'performance_metrics': {
        'accuracy': float(best_result['Accuracy']),
        'precision': float(best_result['Precision']),
        'recall': float(best_result['Recall']),
        'f1_score': float(best_result['F1-Score']),
        'auc_roc': float(best_result['AUC-ROC'])
    },
    'training_details': {
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'n_features': int(X_train.shape[1]),
        'imputation_method': 'KNNImputer (k=5)'
    },
    'feature_names': list(X.columns),
    'target_classes': ['Negative', 'Positive'],
    'class_distribution': {
        'train': {
            'negative': int((y_train == 0).sum()),
            'positive': int((y_train == 1).sum())
        },
        'test': {
            'negative': int((y_test == 0).sum()),
            'positive': int((y_test == 1).sum())
        }
    }
}

metadata_path = os.path.join(models_dir, 'model_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved: {metadata_path}")

# Save full results table
results_csv_path = os.path.join(output_dir, 'all_results.csv')
results_df[display_cols].to_csv(results_csv_path, index=False)
print(f"✓ Full results saved: {results_csv_path}")

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("\n" + "="*80)
print(" ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 Dataset:")
print(f"   - Total samples: {len(df)}")
print(f"   - Features: {X.shape[1]}")
print(f"   - Target: SARS-CoV-2 (Binary: Negative/Positive)")
print(f"   - Class distribution: {(y==0).sum()} Negative, {(y==1).sum()} Positive")

print(f"\n🔧 Preprocessing:")
print(f"   - Missing values imputed: {df.isnull().sum().sum()} → 0 (Hybrid: KNN+MICE)")
print(f"   - Imputation quality: 0.72% avg mean change")
print(f"   - Scaling methods tested: {len(scalers)}")
print(f"   - Train/Test split: 80/20 (stratified)")

print(f"\n🤖 Models Trained:")
print(f"   - Total configurations: {len(all_results)}")
print(f"   - Linear models: Logistic Regression")
print(f"   - KNN: 3 variants (k=3,5,7)")
print(f"   - SVM: 4 kernels (Linear, RBF, Polynomial, Sigmoid)")
print(f"   - Tree-based: Decision Tree, Random Forest")
print(f"   - Boosting: Gradient Boosting, AdaBoost" + (", XGBoost" if XGBOOST_AVAILABLE else "") + (", LightGBM" if LIGHTGBM_AVAILABLE else ""))
print(f"   - Probabilistic: Naive Bayes")

print(f"\n🏆 Best Model:")
print(f"   - Algorithm: {best_result['Model']}")
print(f"   - Scaler: {best_result['Scaler']}")
print(f"   - Accuracy: {best_result['Accuracy']*100:.2f}%")
print(f"   - Precision: {best_result['Precision']*100:.2f}%")
print(f"   - Recall: {best_result['Recall']*100:.2f}%")
print(f"   - F1-Score: {best_result['F1-Score']:.4f}")
print(f"   - AUC-ROC: {best_result['AUC-ROC']:.4f}")

print(f"\n📁 Output Files:")
print(f"   - Models: {models_dir}/")
print(f"   - Plots: {plots_dir}/")
print(f"   - Results: {results_csv_path}")

print("\n" + "="*80)
print("✓ COVID-19 Classification Analysis Complete!")
print("="*80)
