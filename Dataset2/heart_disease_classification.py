import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import xgboost as xgb
import lightgbm as lgb
from joblib import parallel_backend, dump
import warnings
import os

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("MAIN METRIC: ROC-AUC Score (Area Under the ROC Curve)")
print("="*80)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("HEART DISEASE CLASSIFICATION - COMPREHENSIVE ANALYSIS")
print("="*80)

# Create directories for results
for dir_name in ['classification_results', 'classification_results/plots', 
                  'classification_results/gridsearch', 'classification_results/models']:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# --- 1. LOAD AND EXPLORE DATA ---
print("\n1. LOADING AND EXPLORING DATA")
print("=" * 80)

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Dataset2.csv')
print(f"Loading data from: {csv_path}")

df = pd.read_csv(csv_path)
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nDataset Info:")
print(df.info())

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nTarget distribution:")
print(df['target'].value_counts())
print(f"\nTarget balance: {df['target'].value_counts(normalize=True)}")

print(f"\nBasic statistics:")
print(df.describe())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

print(f"\nFeatures: {list(X.columns)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# --- 2. DATA PREPROCESSING ---
print("\n2. DATA PREPROCESSING")
print("=" * 80)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Store unscaled data
X_train_unscaled = X_train.copy()
X_test_unscaled = X_test.copy()

# Scale data for normalized experiments
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

print("Data preprocessing complete.")

# --- 3. DEFINE BASE MODELS ---
print("\n3. DEFINING BASE MODELS")
print("=" * 80)

base_models = {
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'ExtraTrees': ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=1, verbose=-1),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVC_RBF': SVC(kernel='rbf', probability=True, random_state=42),
    'SVC_Poly': SVC(kernel='poly', probability=True, random_state=42, degree=3),
    'SVC_Sigmoid': SVC(kernel='sigmoid', probability=True, random_state=42),
    'SVC_Linear': SVC(kernel='linear', probability=True, random_state=42),
    'LinearSVC': LinearSVC(random_state=42, max_iter=2000)
}

print(f"Defined {len(base_models)} base models:")
for name in base_models.keys():
    print(f"  - {name}")

# --- 4. TRAIN MODELS WITHOUT NORMALIZATION ---
print("\n4. TRAINING MODELS WITHOUT NORMALIZATION")
print("=" * 80)

results_no_norm = []

for model_name, model in base_models.items():
    print(f"\nTraining {model_name}...", end=' ')
    start_time = time.time()
    
    try:
        model.fit(X_train_unscaled, y_train)
        y_pred = model.predict(X_test_unscaled)
        
        # Handle LinearSVC which doesn't have predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_unscaled)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test_unscaled)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred_proba = None
            roc_auc = np.nan
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        elapsed = time.time() - start_time
        
        results_no_norm.append({
            'model': model_name,
            'normalization': 'None',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'training_time_s': elapsed
        })
        
        print(f"AUC={roc_auc:.4f}, Acc={accuracy:.4f}, Time={elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {str(e)}")

# --- 5. TRAIN MODELS WITH STANDARDSCALER ---
print("\n5. TRAINING MODELS WITH STANDARDSCALER")
print("=" * 80)

results_standard = []

for model_name, model_class in base_models.items():
    print(f"\nTraining {model_name}...", end=' ')
    start_time = time.time()
    
    try:
        # Create fresh model instance
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_name == 'ExtraTrees':
            model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=1, verbose=-1)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == 'SVC_RBF':
            model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_name == 'SVC_Poly':
            model = SVC(kernel='poly', probability=True, random_state=42, degree=3)
        elif model_name == 'SVC_Sigmoid':
            model = SVC(kernel='sigmoid', probability=True, random_state=42)
        elif model_name == 'SVC_Linear':
            model = SVC(kernel='linear', probability=True, random_state=42)
        elif model_name == 'LinearSVC':
            model = LinearSVC(random_state=42, max_iter=2000)
        
        model.fit(X_train_standard, y_train)
        y_pred = model.predict(X_test_standard)
        
        # Handle LinearSVC which doesn't have predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_standard)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test_standard)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred_proba = None
            roc_auc = np.nan
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        elapsed = time.time() - start_time
        
        results_standard.append({
            'model': model_name,
            'normalization': 'StandardScaler',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'training_time_s': elapsed
        })
        
        print(f"AUC={roc_auc:.4f}, Acc={accuracy:.4f}, Time={elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {str(e)}")

# --- 6. TRAIN MODELS WITH MINMAXSCALER ---
print("\n6. TRAINING MODELS WITH MINMAXSCALER")
print("=" * 80)

results_minmax = []

for model_name, model_class in base_models.items():
    print(f"\nTraining {model_name}...", end=' ')
    start_time = time.time()
    
    try:
        # Create fresh model instance
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')
        elif model_name == 'RandomForest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        elif model_name == 'GradientBoosting':
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif model_name == 'ExtraTrees':
            model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1)
        elif model_name == 'LightGBM':
            model = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=1, verbose=-1)
        elif model_name == 'LogisticRegression':
            model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_name == 'DecisionTree':
            model = DecisionTreeClassifier(random_state=42)
        elif model_name == 'KNN':
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == 'SVC_RBF':
            model = SVC(kernel='rbf', probability=True, random_state=42)
        elif model_name == 'SVC_Poly':
            model = SVC(kernel='poly', probability=True, random_state=42, degree=3)
        elif model_name == 'SVC_Sigmoid':
            model = SVC(kernel='sigmoid', probability=True, random_state=42)
        elif model_name == 'SVC_Linear':
            model = SVC(kernel='linear', probability=True, random_state=42)
        elif model_name == 'LinearSVC':
            model = LinearSVC(random_state=42, max_iter=2000)
        
        model.fit(X_train_minmax, y_train)
        y_pred = model.predict(X_test_minmax)
        
        # Handle LinearSVC which doesn't have predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test_minmax)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        elif hasattr(model, 'decision_function'):
            y_pred_proba = model.decision_function(X_test_minmax)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            y_pred_proba = None
            roc_auc = np.nan
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        elapsed = time.time() - start_time
        
        results_minmax.append({
            'model': model_name,
            'normalization': 'MinMaxScaler',
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'training_time_s': elapsed
        })
        
        print(f"AUC={roc_auc:.4f}, Acc={accuracy:.4f}, Time={elapsed:.2f}s")
    except Exception as e:
        print(f"Error: {str(e)}")

# Combine all results
all_results = results_no_norm + results_standard + results_minmax
results_df = pd.DataFrame(all_results)

# Save individual model results
results_df.to_csv('classification_results/individual_models_comparison.csv', index=False)
print("\n✓ Individual model results saved to 'classification_results/individual_models_comparison.csv'")

print("\nTop 10 Models by ROC-AUC Score:")
print(results_df.sort_values('roc_auc', ascending=False).head(10)[
    ['model', 'normalization', 'roc_auc', 'accuracy', 'f1_score']
].to_string())

# --- 7. GRIDSEARCH ON TOP MODELS ---
print("\n7. GRIDSEARCHCV ON TOP MODELS")
print("=" * 80)

# Select top unique models for GridSearch
top_models = (
    results_df.sort_values('roc_auc', ascending=False)
    .drop_duplicates(subset=['model'], keep='first')
    .head(5)
)
print(f"\nPerforming GridSearch on top {len(top_models)} unique models (by ROC-AUC):")
for idx, row in top_models.iterrows():
    print(f"  - {row['model']} (normalization={row['normalization']}, AUC={row['roc_auc']:.4f})")

gridsearch_results = []

# GridSearch for XGBoost (More comprehensive)
print("\n--- GridSearch: XGBoost (Comprehensive) ---")
xgb_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

xgb_grid = GridSearchCV(
    xgb.XGBClassifier(random_state=42, n_jobs=1, eval_metric='logloss'),
    xgb_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

print("Fitting XGBoost GridSearch...")
start_time = time.time()
xgb_grid.fit(X_train_standard, y_train)
xgb_grid_time = time.time() - start_time

print(f"Best XGBoost params: {xgb_grid.best_params_}")
print(f"Best CV ROC-AUC score: {xgb_grid.best_score_:.4f}")

y_pred_xgb_grid = xgb_grid.predict(X_test_standard)
y_pred_proba_xgb_grid = xgb_grid.predict_proba(X_test_standard)[:, 1]

gridsearch_results.append({
    'model': 'XGBoost_GridSearch',
    'best_params': str(xgb_grid.best_params_),
    'accuracy': accuracy_score(y_test, y_pred_xgb_grid),
    'precision': precision_score(y_test, y_pred_xgb_grid),
    'recall': recall_score(y_test, y_pred_xgb_grid),
    'f1_score': f1_score(y_test, y_pred_xgb_grid),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_xgb_grid),
    'training_time_s': xgb_grid_time
})

print(f"Test ROC-AUC: {gridsearch_results[-1]['roc_auc']:.4f}, Accuracy: {gridsearch_results[-1]['accuracy']:.4f}")

# GridSearch for KNN (NEW)
print("\n--- GridSearch: KNN (NEW) ---")
knn_params = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

knn_grid = GridSearchCV(
    KNeighborsClassifier(),
    knn_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

print("Fitting KNN GridSearch...")
start_time = time.time()
knn_grid.fit(X_train_standard, y_train)
knn_grid_time = time.time() - start_time

print(f"Best KNN params: {knn_grid.best_params_}")
print(f"Best CV ROC-AUC score: {knn_grid.best_score_:.4f}")

y_pred_knn_grid = knn_grid.predict(X_test_standard)
y_pred_proba_knn_grid = knn_grid.predict_proba(X_test_standard)[:, 1]

gridsearch_results.append({
    'model': 'KNN_GridSearch',
    'best_params': str(knn_grid.best_params_),
    'accuracy': accuracy_score(y_test, y_pred_knn_grid),
    'precision': precision_score(y_test, y_pred_knn_grid),
    'recall': recall_score(y_test, y_pred_knn_grid),
    'f1_score': f1_score(y_test, y_pred_knn_grid),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_knn_grid),
    'training_time_s': knn_grid_time
})

print(f"Test ROC-AUC: {gridsearch_results[-1]['roc_auc']:.4f}, Accuracy: {gridsearch_results[-1]['accuracy']:.4f}")

# GridSearch for RandomForest
print("\n--- GridSearch: RandomForest ---")
rf_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=1),
    rf_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

print("Fitting RandomForest GridSearch...")
start_time = time.time()
rf_grid.fit(X_train_standard, y_train)
rf_grid_time = time.time() - start_time

print(f"Best RandomForest params: {rf_grid.best_params_}")
print(f"Best CV ROC-AUC score: {rf_grid.best_score_:.4f}")

y_pred_rf_grid = rf_grid.predict(X_test_standard)
y_pred_proba_rf_grid = rf_grid.predict_proba(X_test_standard)[:, 1]

gridsearch_results.append({
    'model': 'RandomForest_GridSearch',
    'best_params': str(rf_grid.best_params_),
    'accuracy': accuracy_score(y_test, y_pred_rf_grid),
    'precision': precision_score(y_test, y_pred_rf_grid),
    'recall': recall_score(y_test, y_pred_rf_grid),
    'f1_score': f1_score(y_test, y_pred_rf_grid),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_rf_grid),
    'training_time_s': rf_grid_time
})

print(f"Test ROC-AUC: {gridsearch_results[-1]['roc_auc']:.4f}, Accuracy: {gridsearch_results[-1]['accuracy']:.4f}")

# GridSearch for GradientBoosting
print("\n--- GridSearch: GradientBoosting ---")
gb_params = {
    'n_estimators': [50, 100, 150, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0]
}

gb_grid = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    gb_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=1,
    verbose=1
)

print("Fitting GradientBoosting GridSearch...")
start_time = time.time()
gb_grid.fit(X_train_standard, y_train)
gb_grid_time = time.time() - start_time

print(f"Best GradientBoosting params: {gb_grid.best_params_}")
print(f"Best CV ROC-AUC score: {gb_grid.best_score_:.4f}")

y_pred_gb_grid = gb_grid.predict(X_test_standard)
y_pred_proba_gb_grid = gb_grid.predict_proba(X_test_standard)[:, 1]

gridsearch_results.append({
    'model': 'GradientBoosting_GridSearch',
    'best_params': str(gb_grid.best_params_),
    'accuracy': accuracy_score(y_test, y_pred_gb_grid),
    'precision': precision_score(y_test, y_pred_gb_grid),
    'recall': recall_score(y_test, y_pred_gb_grid),
    'f1_score': f1_score(y_test, y_pred_gb_grid),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_gb_grid),
    'training_time_s': gb_grid_time
})

print(f"Test ROC-AUC: {gridsearch_results[-1]['roc_auc']:.4f}, Accuracy: {gridsearch_results[-1]['accuracy']:.4f}")

gridsearch_df = pd.DataFrame(gridsearch_results)
gridsearch_df.to_csv('classification_results/gridsearch/gridsearch_results.csv', index=False)
print("\n✓ GridSearch results saved to 'classification_results/gridsearch/gridsearch_results.csv'")

print("\nGridSearch Results (sorted by ROC-AUC):")
print(gridsearch_df.sort_values('roc_auc', ascending=False)[['model', 'roc_auc', 'accuracy', 'f1_score']].to_string())

# --- 8. ENSEMBLE METHODS WITHOUT NORMALIZATION ---
print("\n8. ENSEMBLE METHODS WITHOUT NORMALIZATION")
print("=" * 80)

ensemble_results = []

# Voting Classifier
print("\n--- Voting Classifier (No Normalization) ---")
voting_estimators_no_norm = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1))
]

print("Training Voting Classifier...")
start_time = time.time()
voting_clf_no_norm = VotingClassifier(estimators=voting_estimators_no_norm, voting='soft', n_jobs=1)
with parallel_backend('threading'):
    voting_clf_no_norm.fit(X_train_unscaled, y_train)
    y_pred_voting_no_norm = voting_clf_no_norm.predict(X_test_unscaled)
    y_pred_proba_voting_no_norm = voting_clf_no_norm.predict_proba(X_test_unscaled)[:, 1]
voting_no_norm_time = time.time() - start_time

ensemble_results.append({
    'model': 'VotingClassifier',
    'normalization': 'None',
    'accuracy': accuracy_score(y_test, y_pred_voting_no_norm),
    'precision': precision_score(y_test, y_pred_voting_no_norm),
    'recall': recall_score(y_test, y_pred_voting_no_norm),
    'f1_score': f1_score(y_test, y_pred_voting_no_norm),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_voting_no_norm),
    'training_time_s': voting_no_norm_time
})

print(f"Voting (No Norm) - Acc={ensemble_results[-1]['accuracy']:.4f}, F1={ensemble_results[-1]['f1_score']:.4f}")

# Stacking Classifier
print("\n--- Stacking Classifier (No Normalization) ---")
stacking_estimators_no_norm = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1))
]

print("Training Stacking Classifier...")
start_time = time.time()
stacking_clf_no_norm = StackingClassifier(
    estimators=stacking_estimators_no_norm,
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5,
    n_jobs=1
)
stacking_clf_no_norm.fit(X_train_unscaled, y_train)
y_pred_stacking_no_norm = stacking_clf_no_norm.predict(X_test_unscaled)
y_pred_proba_stacking_no_norm = stacking_clf_no_norm.predict_proba(X_test_unscaled)[:, 1]
stacking_no_norm_time = time.time() - start_time

ensemble_results.append({
    'model': 'StackingClassifier',
    'normalization': 'None',
    'accuracy': accuracy_score(y_test, y_pred_stacking_no_norm),
    'precision': precision_score(y_test, y_pred_stacking_no_norm),
    'recall': recall_score(y_test, y_pred_stacking_no_norm),
    'f1_score': f1_score(y_test, y_pred_stacking_no_norm),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_stacking_no_norm),
    'training_time_s': stacking_no_norm_time
})

print(f"Stacking (No Norm) - Acc={ensemble_results[-1]['accuracy']:.4f}, F1={ensemble_results[-1]['f1_score']:.4f}")

# --- 9. ENSEMBLE METHODS WITH STANDARDSCALER ---
print("\n9. ENSEMBLE METHODS WITH STANDARDSCALER")
print("=" * 80)

# Voting Classifier
print("\n--- Voting Classifier (StandardScaler) ---")
voting_estimators_standard = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1))
]

print("Training Voting Classifier...")
start_time = time.time()
voting_clf_standard = VotingClassifier(estimators=voting_estimators_standard, voting='soft', n_jobs=1)
with parallel_backend('threading'):
    voting_clf_standard.fit(X_train_standard, y_train)
    y_pred_voting_standard = voting_clf_standard.predict(X_test_standard)
    y_pred_proba_voting_standard = voting_clf_standard.predict_proba(X_test_standard)[:, 1]
voting_standard_time = time.time() - start_time

ensemble_results.append({
    'model': 'VotingClassifier',
    'normalization': 'StandardScaler',
    'accuracy': accuracy_score(y_test, y_pred_voting_standard),
    'precision': precision_score(y_test, y_pred_voting_standard),
    'recall': recall_score(y_test, y_pred_voting_standard),
    'f1_score': f1_score(y_test, y_pred_voting_standard),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_voting_standard),
    'training_time_s': voting_standard_time
})

print(f"Voting (Standard) - Acc={ensemble_results[-1]['accuracy']:.4f}, F1={ensemble_results[-1]['f1_score']:.4f}")

# Stacking Classifier
print("\n--- Stacking Classifier (StandardScaler) ---")
stacking_estimators_standard = [
    ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=1))
]

print("Training Stacking Classifier...")
start_time = time.time()
stacking_clf_standard = StackingClassifier(
    estimators=stacking_estimators_standard,
    final_estimator=LogisticRegression(random_state=42, max_iter=1000),
    cv=5,
    n_jobs=1
)
stacking_clf_standard.fit(X_train_standard, y_train)
y_pred_stacking_standard = stacking_clf_standard.predict(X_test_standard)
y_pred_proba_stacking_standard = stacking_clf_standard.predict_proba(X_test_standard)[:, 1]
stacking_standard_time = time.time() - start_time

ensemble_results.append({
    'model': 'StackingClassifier',
    'normalization': 'StandardScaler',
    'accuracy': accuracy_score(y_test, y_pred_stacking_standard),
    'precision': precision_score(y_test, y_pred_stacking_standard),
    'recall': recall_score(y_test, y_pred_stacking_standard),
    'f1_score': f1_score(y_test, y_pred_stacking_standard),
    'roc_auc': roc_auc_score(y_test, y_pred_proba_stacking_standard),
    'training_time_s': stacking_standard_time
})

print(f"Stacking (Standard) - Acc={ensemble_results[-1]['accuracy']:.4f}, F1={ensemble_results[-1]['f1_score']:.4f}")

# Save ensemble results
ensemble_df = pd.DataFrame(ensemble_results)
ensemble_df.to_csv('classification_results/ensemble_methods_comparison.csv', index=False)
print("\n✓ Ensemble results saved to 'classification_results/ensemble_methods_comparison.csv'")

print("\nEnsemble Methods Results:")
print(ensemble_df[['model', 'normalization', 'accuracy', 'f1_score', 'roc_auc']].to_string())

# --- 10. FINAL COMPARISON AND VISUALIZATION ---
print("\n10. FINAL COMPARISON AND SAVE BEST MODEL")
print("=" * 80)

# Combine all results
final_results = pd.concat([results_df, gridsearch_df.drop('best_params', axis=1), ensemble_df], ignore_index=True)
final_results = final_results.sort_values('roc_auc', ascending=False)

# Save final results
final_results.to_csv('classification_results/final_comparison_all_methods.csv', index=False)
print("\n✓ Final comparison saved to 'classification_results/final_comparison_all_methods.csv'")

print("\nTop 10 Best Models by ROC-AUC Score:")
print(final_results.head(10)[['model', 'normalization', 'roc_auc', 'accuracy', 'f1_score']].to_string())

# --- SAVE BEST MODEL ---
print("\n" + "=" * 80)
print("SAVING BEST MODEL FOR FUTURE PREDICTIONS")
print("=" * 80)

best_model_row = final_results.iloc[0]
print(f"\nBest Model: {best_model_row['model']}")
print(f"Normalization: {best_model_row['normalization']}")
print(f"ROC-AUC: {best_model_row['roc_auc']:.4f}")
print(f"Accuracy: {best_model_row['accuracy']:.4f}")
print(f"F1-Score: {best_model_row['f1_score']:.4f}")

# Retrain best model on full dataset or get from GridSearch
best_model_name = best_model_row['model']
best_normalization = best_model_row['normalization']

# Determine which scaler to use
if pd.notna(best_normalization):
    if best_normalization == 'StandardScaler':
        scaler_to_save = scaler_standard
        X_train_final = X_train_standard
        X_test_final = X_test_standard
    elif best_normalization == 'MinMaxScaler':
        scaler_to_save = scaler_minmax
        X_train_final = X_train_minmax
        X_test_final = X_test_minmax
    else:
        scaler_to_save = None
        X_train_final = X_train_unscaled
        X_test_final = X_test_unscaled
else:
    scaler_to_save = scaler_standard  # GridSearch used StandardScaler
    X_train_final = X_train_standard
    X_test_final = X_test_standard

# Get or retrain the best model
if 'XGBoost_GridSearch' in best_model_name:
    best_model_trained = xgb_grid.best_estimator_
    y_pred_best_model = y_pred_xgb_grid
    y_pred_proba_best_model = y_pred_proba_xgb_grid
elif 'KNN_GridSearch' in best_model_name:
    best_model_trained = knn_grid.best_estimator_
    y_pred_best_model = y_pred_knn_grid
    y_pred_proba_best_model = y_pred_proba_knn_grid
elif 'RandomForest_GridSearch' in best_model_name:
    best_model_trained = rf_grid.best_estimator_
    y_pred_best_model = y_pred_rf_grid
    y_pred_proba_best_model = y_pred_proba_rf_grid
elif 'GradientBoosting_GridSearch' in best_model_name:
    best_model_trained = gb_grid.best_estimator_
    y_pred_best_model = y_pred_gb_grid
    y_pred_proba_best_model = y_pred_proba_gb_grid
elif 'VotingClassifier' in best_model_name:
    if best_normalization == 'None':
        best_model_trained = voting_clf_no_norm
        y_pred_best_model = y_pred_voting_no_norm
        y_pred_proba_best_model = y_pred_proba_voting_no_norm
    else:
        best_model_trained = voting_clf_standard
        y_pred_best_model = y_pred_voting_standard
        y_pred_proba_best_model = y_pred_proba_voting_standard
elif 'StackingClassifier' in best_model_name:
    if best_normalization == 'None':
        best_model_trained = stacking_clf_no_norm
        y_pred_best_model = y_pred_stacking_no_norm
        y_pred_proba_best_model = y_pred_proba_stacking_no_norm
    else:
        best_model_trained = stacking_clf_standard
        y_pred_best_model = y_pred_stacking_standard
        y_pred_proba_best_model = y_pred_proba_stacking_standard
else:
    # Retrain individual model
    print(f"\nRetraining {best_model_name} with best configuration...")
    if best_model_name == 'XGBoost':
        best_model_trained = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')
    elif best_model_name == 'RandomForest':
        best_model_trained = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    elif best_model_name == 'GradientBoosting':
        best_model_trained = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif best_model_name == 'LightGBM':
        best_model_trained = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=1, verbose=-1)
    elif best_model_name == 'SVC_RBF':
        best_model_trained = SVC(kernel='rbf', probability=True, random_state=42)
    elif best_model_name == 'SVC_Poly':
        best_model_trained = SVC(kernel='poly', probability=True, random_state=42, degree=3)
    elif best_model_name == 'SVC_Sigmoid':
        best_model_trained = SVC(kernel='sigmoid', probability=True, random_state=42)
    elif best_model_name == 'SVC_Linear':
        best_model_trained = SVC(kernel='linear', probability=True, random_state=42)
    elif best_model_name == 'KNN':
        best_model_trained = KNeighborsClassifier(n_neighbors=5)
    elif best_model_name == 'LogisticRegression':
        best_model_trained = LogisticRegression(random_state=42, max_iter=1000)
    else:
        best_model_trained = base_models.get(best_model_name)
    
    best_model_trained.fit(X_train_final, y_train)
    y_pred_best_model = best_model_trained.predict(X_test_final)
    if hasattr(best_model_trained, 'predict_proba'):
        y_pred_proba_best_model = best_model_trained.predict_proba(X_test_final)[:, 1]
    elif hasattr(best_model_trained, 'decision_function'):
        y_pred_proba_best_model = best_model_trained.decision_function(X_test_final)
    else:
        y_pred_proba_best_model = None

# Save the model and scaler
dump(best_model_trained, 'classification_results/models/best_model.joblib')
print(f"✓ Best model saved to 'classification_results/models/best_model.joblib'")

if scaler_to_save is not None:
    dump(scaler_to_save, 'classification_results/models/scaler.joblib')
    print(f"✓ Scaler saved to 'classification_results/models/scaler.joblib'")
else:
    print("✓ No scaler needed (model works on raw features)")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'normalization': best_normalization if pd.notna(best_normalization) else 'None',
    'roc_auc': float(best_model_row['roc_auc']),
    'accuracy': float(best_model_row['accuracy']),
    'precision': float(best_model_row['precision']),
    'recall': float(best_model_row['recall']),
    'f1_score': float(best_model_row['f1_score']),
    'training_time_s': float(best_model_row['training_time_s']),
    'feature_names': list(X.columns),
    'target_name': 'target'
}

import json
with open('classification_results/models/model_metadata.json', 'w') as f:
    json.dump(model_metadata, f, indent=4)
print("✓ Model metadata saved to 'classification_results/models/model_metadata.json'")

print("\n" + "=" * 80)
print("BEST MODEL IS READY FOR PREDICTIONS IN THE DASHBOARD!")
print("=" * 80)

# Create separate, readable visualizations
print("\nGenerating visualizations...")

# PLOT 1: Top Models Comparison
print("  - Creating top models comparison plot...")
fig1, axes1 = plt.subplots(2, 2, figsize=(20, 14))
fig1.suptitle('Top Models Performance Comparison (Primary Metric: ROC-AUC)', fontsize=18, fontweight='bold', y=0.995)

# 1. Top 10 Models by ROC-AUC
ax1 = axes1[0, 0]
top_10 = final_results.head(10).copy()
top_10['short_label'] = top_10.apply(lambda x: f"{x['model'][:12]}\n({x['normalization'] if pd.notna(x['normalization']) else 'Grid'})", axis=1)
colors_top10 = ['green' if 'Voting' in m or 'Stacking' in m 
                else 'orange' if 'GridSearch' in m else 'lightblue' 
                for m in top_10['model']]
bars = ax1.barh(range(len(top_10)), top_10['roc_auc'], color=colors_top10, edgecolor='black', linewidth=1.5)
ax1.set_yticks(range(len(top_10)))
ax1.set_yticklabels(top_10['short_label'], fontsize=11)
ax1.set_xlabel('ROC-AUC Score', fontsize=13, fontweight='bold')
ax1.set_title('Top 10 Models by ROC-AUC (PRIMARY METRIC)', fontsize=14, fontweight='bold', color='darkred')
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()
# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_10['roc_auc'])):
    ax1.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

# 2. Accuracy Comparison
ax2 = axes1[0, 1]
top_10_acc = final_results.head(10).copy()
bars2 = ax2.barh(range(len(top_10_acc)), top_10_acc['accuracy'], color=colors_top10, edgecolor='black', linewidth=1.5)
ax2.set_yticks(range(len(top_10_acc)))
ax2.set_yticklabels(top_10['short_label'], fontsize=11)
ax2.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
ax2.set_title('Top 10 Models by Accuracy', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars2, top_10_acc['accuracy'])):
    ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

# 3. ROC-AUC Comparison
ax3 = axes1[1, 0]
top_10_roc = final_results.head(10).copy()
bars3 = ax3.barh(range(len(top_10_roc)), top_10_roc['roc_auc'], color=colors_top10, edgecolor='black', linewidth=1.5)
ax3.set_yticks(range(len(top_10_roc)))
ax3.set_yticklabels(top_10['short_label'], fontsize=11)
ax3.set_xlabel('ROC-AUC Score', fontsize=13, fontweight='bold')
ax3.set_title('Top 10 Models by ROC-AUC', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3, linestyle='--')
ax3.invert_yaxis()
for i, (bar, val) in enumerate(zip(bars3, top_10_roc['roc_auc'])):
    ax3.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

# 4. Precision vs Recall
ax4 = axes1[1, 1]
top_10_pr = final_results.head(10)
scatter = ax4.scatter(top_10_pr['recall'], top_10_pr['precision'], 
           s=top_10_pr['accuracy']*1000, alpha=0.6, c=range(len(top_10_pr)), 
           cmap='viridis', edgecolors='black', linewidths=2)
for idx, row in top_10_pr.iterrows():
    ax4.annotate(row['model'][:10], (row['recall'], row['precision']), 
                fontsize=9, fontweight='bold', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
ax4.set_xlabel('Recall', fontsize=13, fontweight='bold')
ax4.set_ylabel('Precision', fontsize=13, fontweight='bold')
ax4.set_title('Precision vs Recall (Size = Accuracy)', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3, linestyle='--')
plt.colorbar(scatter, ax=ax4, label='Rank')

plt.tight_layout()
plt.savefig('classification_results/plots/01_top_models_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 2: Normalization Impact
print("  - Creating normalization impact plot...")
fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8))
fig2.suptitle('Normalization Impact Analysis', fontsize=18, fontweight='bold')

# Normalization Impact on Individual Models
ax5 = axes2[0]
norm_comparison = results_df.groupby(['model', 'normalization'])['accuracy'].mean().reset_index()
norm_pivot = norm_comparison.pivot(index='model', columns='normalization', values='accuracy')
norm_pivot.plot(kind='barh', ax=ax5, width=0.7, edgecolor='black', linewidth=1.2)
ax5.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
ax5.set_ylabel('Model', fontsize=13, fontweight='bold')
ax5.set_title('Normalization Impact on Individual Models', fontsize=14, fontweight='bold')
ax5.legend(title='Normalization', fontsize=11, title_fontsize=12)
ax5.grid(axis='x', alpha=0.3, linestyle='--')
ax5.tick_params(axis='both', labelsize=11)

# Normalization Effect Summary
ax6 = axes2[1]
norm_effect = results_df.groupby('normalization')['accuracy'].mean().sort_values(ascending=False)
bars_norm = ax6.bar(norm_effect.index, norm_effect.values, 
        color=['lightgreen', 'lightskyblue', 'lightcoral'], 
        alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('Average Accuracy', fontsize=13, fontweight='bold')
ax6.set_xlabel('Normalization Method', fontsize=13, fontweight='bold')
ax6.set_title('Normalization Effect (Average Across All Models)', fontsize=14, fontweight='bold')
ax6.grid(axis='y', alpha=0.3, linestyle='--')
ax6.tick_params(axis='both', labelsize=11)
# Add value labels
for bar, val in zip(bars_norm, norm_effect.values):
    ax6.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('classification_results/plots/02_normalization_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 3: Ensemble and GridSearch Analysis
print("  - Creating ensemble and GridSearch analysis plot...")
fig3, axes3 = plt.subplots(2, 2, figsize=(20, 14))
fig3.suptitle('Ensemble Methods and GridSearch Analysis', fontsize=18, fontweight='bold')

# Ensemble vs Individual Best
ax7 = axes3[0, 0]
best_individual = results_df.sort_values('accuracy', ascending=False).iloc[0]
ensemble_only = ensemble_df.copy()
comparison_data = pd.DataFrame([
    {'Type': 'Best Individual', 'Accuracy': best_individual['accuracy'], 'Method': 'Individual'},
    {'Type': 'Voting\n(No Norm)', 'Accuracy': ensemble_only[(ensemble_only['model']=='VotingClassifier') & (ensemble_only['normalization']=='None')]['accuracy'].values[0], 'Method': 'Ensemble'},
    {'Type': 'Voting\n(Standard)', 'Accuracy': ensemble_only[(ensemble_only['model']=='VotingClassifier') & (ensemble_only['normalization']=='StandardScaler')]['accuracy'].values[0], 'Method': 'Ensemble'},
    {'Type': 'Stacking\n(No Norm)', 'Accuracy': ensemble_only[(ensemble_only['model']=='StackingClassifier') & (ensemble_only['normalization']=='None')]['accuracy'].values[0], 'Method': 'Ensemble'},
    {'Type': 'Stacking\n(Standard)', 'Accuracy': ensemble_only[(ensemble_only['model']=='StackingClassifier') & (ensemble_only['normalization']=='StandardScaler')]['accuracy'].values[0], 'Method': 'Ensemble'}
])
colors_ens = ['blue', 'green', 'green', 'orange', 'orange']
bars_ens = ax7.bar(comparison_data['Type'], comparison_data['Accuracy'], 
        color=colors_ens, alpha=0.7, edgecolor='black', linewidth=2)
ax7.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax7.set_xlabel('Model Type', fontsize=13, fontweight='bold')
ax7.set_title('Ensemble Methods vs Best Individual Model', fontsize=14, fontweight='bold')
ax7.axhline(y=best_individual['accuracy'], color='red', linestyle='--', alpha=0.7, linewidth=2, label='Best Individual Baseline')
ax7.legend(fontsize=11)
ax7.grid(axis='y', alpha=0.3, linestyle='--')
ax7.tick_params(axis='both', labelsize=11)
for bar, val in zip(bars_ens, comparison_data['Accuracy']):
    ax7.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
            f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# GridSearch Improvements
ax8 = axes3[0, 1]
gridsearch_comparison = []
for _, row in gridsearch_df.iterrows():
    base_name = row['model'].replace('_GridSearch', '')
    base_acc = results_df[results_df['model']==base_name]['accuracy'].max()
    grid_acc = row['accuracy']
    improvement = ((grid_acc - base_acc) / base_acc) * 100
    gridsearch_comparison.append({
        'Model': base_name,
        'Improvement': improvement,
        'Base': base_acc,
        'GridSearch': grid_acc
    })
gs_comp_df = pd.DataFrame(gridsearch_comparison)
colors_gs = ['green' if v > 0 else 'red' for v in gs_comp_df['Improvement']]
bars_gs = ax8.bar(gs_comp_df['Model'], gs_comp_df['Improvement'], color=colors_gs, alpha=0.7, edgecolor='black', linewidth=2)
ax8.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
ax8.set_xlabel('Model', fontsize=13, fontweight='bold')
ax8.set_title('GridSearch Improvement over Base Models', fontsize=14, fontweight='bold')
ax8.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax8.grid(axis='y', alpha=0.3, linestyle='--')
ax8.tick_params(axis='both', labelsize=11)
for bar, val in zip(bars_gs, gs_comp_df['Improvement']):
    ax8.text(bar.get_x() + bar.get_width()/2, val + 0.1 if val > 0 else val - 0.3, 
            f'{val:.2f}%', ha='center', va='bottom' if val > 0 else 'top', 
            fontsize=11, fontweight='bold')

# Training Time Comparison
ax9 = axes3[1, 0]
time_comparison = final_results.head(10).copy()
time_comparison['short_label'] = time_comparison.apply(lambda x: f"{x['model'][:15]}", axis=1)
bars_time = ax9.barh(range(len(time_comparison)), time_comparison['training_time_s'], 
                     color='coral', alpha=0.7, edgecolor='black', linewidth=1.5)
ax9.set_yticks(range(len(time_comparison)))
ax9.set_yticklabels(time_comparison['short_label'], fontsize=11)
ax9.set_xlabel('Training Time (seconds)', fontsize=13, fontweight='bold')
ax9.set_title('Training Time - Top 10 Models', fontsize=14, fontweight='bold')
ax9.grid(axis='x', alpha=0.3, linestyle='--')
ax9.invert_yaxis()
ax9.tick_params(axis='both', labelsize=11)
for i, (bar, val) in enumerate(zip(bars_time, time_comparison['training_time_s'])):
    ax9.text(val + 0.5, i, f'{val:.2f}s', va='center', fontsize=10, fontweight='bold')

# Model Type Distribution
ax10 = axes3[1, 1]
model_types = []
for model in final_results['model']:
    if 'Voting' in model or 'Stacking' in model:
        model_types.append('Ensemble')
    elif 'GridSearch' in model:
        model_types.append('GridSearch')
    else:
        model_types.append('Individual')
type_counts = pd.Series(model_types).value_counts()
wedges, texts, autotexts = ax10.pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%', 
       colors=['lightblue', 'orange', 'lightgreen'], startangle=90,
       textprops={'fontsize': 13, 'fontweight': 'bold'})
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
ax10.set_title('Model Type Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('classification_results/plots/03_ensemble_gridsearch_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# PLOT 4: Best Model Details
print("  - Creating best model details plot...")
fig4, axes4 = plt.subplots(1, 2, figsize=(18, 8))
fig4.suptitle('Best Model Performance Details', fontsize=18, fontweight='bold')

# Confusion Matrix for Best Model
ax11 = axes4[0]
best_model = final_results.iloc[0]
print(f"    Generating confusion matrix for: {best_model['model']}")

# Get predictions for best model
if 'GridSearch' in best_model['model']:
    if 'XGBoost' in best_model['model']:
        y_pred_best = y_pred_xgb_grid
    elif 'RandomForest' in best_model['model']:
        y_pred_best = y_pred_rf_grid
    elif 'GradientBoosting' in best_model['model']:
        y_pred_best = y_pred_gb_grid
elif best_model['model'] == 'VotingClassifier':
    if best_model['normalization'] == 'None':
        y_pred_best = y_pred_voting_no_norm
    else:
        y_pred_best = y_pred_voting_standard
elif best_model['model'] == 'StackingClassifier':
    if best_model['normalization'] == 'None':
        y_pred_best = y_pred_stacking_no_norm
    else:
        y_pred_best = y_pred_stacking_standard
else:
    # Retrain best individual model to get predictions
    best_model_name = best_model['model']
    if best_model['normalization'] == 'StandardScaler':
        X_train_use, X_test_use = X_train_standard, X_test_standard
    elif best_model['normalization'] == 'MinMaxScaler':
        X_train_use, X_test_use = X_train_minmax, X_test_minmax
    else:
        X_train_use, X_test_use = X_train_unscaled, X_test_unscaled
    
    if best_model_name == 'XGBoost':
        best_m = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=1, eval_metric='logloss')
    elif best_model_name == 'RandomForest':
        best_m = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    elif best_model_name == 'LightGBM':
        best_m = lgb.LGBMClassifier(n_estimators=100, random_state=42, n_jobs=1, verbose=-1)
    else:
        best_m = base_models[best_model_name]
    
    best_m.fit(X_train_use, y_train)
    y_pred_best = best_m.predict(X_test_use)

cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax11, cbar=True,
           annot_kws={'fontsize': 16, 'fontweight': 'bold'},
           cbar_kws={'label': 'Count'})
ax11.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
ax11.set_ylabel('True Label', fontsize=13, fontweight='bold')
ax11.set_title(f'Confusion Matrix - {best_model["model"][:25]}', fontsize=14, fontweight='bold')
ax11.tick_params(axis='both', labelsize=12)

# Summary Statistics
ax12 = axes4[1]
ax12.axis('off')
summary_text = f"""
═══════════════════════════════════
    SUMMARY STATISTICS
    PRIMARY METRIC: ROC-AUC
═══════════════════════════════════

Best Overall Model (by ROC-AUC):
  {best_model['model']}
  Normalization: {best_model['normalization'] if pd.notna(best_model['normalization']) else 'GridSearch'}
  
  ROC-AUC:     {best_model['roc_auc']:.4f} ⭐
  Accuracy:    {best_model['accuracy']:.4f}
  Precision:   {best_model['precision']:.4f}
  Recall:      {best_model['recall']:.4f}
  F1 Score:    {best_model['f1_score']:.4f}

───────────────────────────────────
Total Models Evaluated: {len(final_results)}
  • Individual Models:  {len(results_df)}
  • GridSearch Tuned:   {len(gridsearch_df)}
  • Ensemble Methods:   {len(ensemble_df)}

───────────────────────────────────
Best Normalization Method:
  {norm_effect.idxmax()}
  Average ROC-AUC: {final_results.groupby('normalization')['roc_auc'].mean().max():.4f}

───────────────────────────────────
Top 3 Models (by ROC-AUC):

1. {final_results.iloc[0]['model'][:30]}
   ROC-AUC: {final_results.iloc[0]['roc_auc']:.4f}

2. {final_results.iloc[1]['model'][:30]}
   ROC-AUC: {final_results.iloc[1]['roc_auc']:.4f}

3. {final_results.iloc[2]['model'][:30]}
   ROC-AUC: {final_results.iloc[2]['roc_auc']:.4f}

═══════════════════════════════════
"""
ax12.text(0.05, 0.95, summary_text, transform=ax12.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.tight_layout()
plt.savefig('classification_results/plots/04_best_model_details.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ All visualizations generated successfully!")

print("\n=== ANALYSIS COMPLETE ===")
print("Generated files:")
print("- classification_results/individual_models_comparison.csv")
print("- classification_results/gridsearch/gridsearch_results.csv")
print("- classification_results/ensemble_methods_comparison.csv")
print("- classification_results/final_comparison_all_methods.csv")
print("- classification_results/comprehensive_analysis_visualization.png")
print("="*80)
