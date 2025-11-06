import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.ensemble import (
    VotingRegressor, 
    StackingRegressor,
    RandomForestRegressor, 
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from joblib import parallel_backend, dump
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import warnings
import os
import json

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*70)
print("ENSEMBLE METHODS COMPARISON: VOTING vs STACKING")
print("="*70)

# Create a directory for results if it doesn't exist
if not os.path.exists('ensemble_results'):
    os.makedirs('ensemble_results')

# Create models directory for saving best model
if not os.path.exists('ensemble_results/models'):
    os.makedirs('ensemble_results/models')

# --- 1. LOAD AND PREPROCESS DATA ---
print("\n1. LOADING AND PREPROCESSING DATA")
print("=" * 70)

# Load the dataset
df = pd.read_csv('Dataset1.csv')
print(f"Original dataset shape: {df.shape}")

# Drop specified columns
columns_to_drop = ['Apparent Temperature (C)', 'Formatted Date', 'Daily Summary']
df = df.drop(columns=columns_to_drop)
print(f"Dataset shape after dropping columns: {df.shape}")

# Separate features and target
X = df.drop('Temperature (C)', axis=1)
y = df['Temperature (C)']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"Categorical features: {list(categorical_features)}")
print(f"Numerical features: {list(numerical_features)}")

# Encode categorical features
print("\nEncoding categorical features...")
for col in categorical_features:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X[col] = le.fit_transform(X[col])
    try:
        nan_code = list(le.classes_).index('nan')
        X[col] = X[col].replace(nan_code, np.nan)
    except ValueError:
        pass

# Impute missing values using KNNImputer
print("Imputing missing values with KNNImputer...")
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Total missing values after imputation: {X.isnull().sum().sum()}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete.")
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# --- 2. DEFINE BASE MODELS ---
print("\n2. DEFINING BASE MODELS")
print("=" * 70)

# Define diverse base models for ensemble
base_models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'RandomForest': RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
    'ExtraTrees': ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1),
    'Ridge': Ridge(alpha=1.0, random_state=42),
    'Lasso': Lasso(alpha=0.1, random_state=42),
    'ElasticNet': ElasticNet(alpha=0.1, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
}

print(f"Defined {len(base_models)} base models:")
for name in base_models.keys():
    print(f"  - {name}")

# --- 3. EVALUATE INDIVIDUAL MODELS ---
print("\n3. EVALUATING INDIVIDUAL BASE MODELS")
print("=" * 70)

individual_results = []

for model_name, model in base_models.items():
    print(f"\nTraining {model_name}...", end=' ')
    start_time = time.time()
    
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)
    elapsed = time.time() - start_time
    
    individual_results.append({
        'model': model_name,
        'ensemble_type': 'Individual',
        'r2_score': r2,
        'mse': mse,
        'mae': mae,
        'explained_variance': ev,
        'training_time_s': elapsed
    })
    
    print(f"R²={r2:.4f}, MSE={mse:.4f}, Time={elapsed:.2f}s")

# Save individual results
individual_df = pd.DataFrame(individual_results).sort_values('r2_score', ascending=False)
print("\nIndividual Model Rankings:")
print(individual_df[['model', 'r2_score', 'mse', 'mae']].to_string())

# --- 4. VOTING REGRESSOR (HETEROGENEOUS PARALLEL ENSEMBLE) ---
print("\n4. VOTING REGRESSOR - HETEROGENEOUS PARALLEL ENSEMBLE")
print("=" * 70)

print("\nVoting combines predictions from multiple diverse models in parallel.")
print("Each model makes independent predictions, then they are averaged.")

# Create voting ensemble with top performing models
# Use fewer parallel workers and thread-based backend to avoid heavy pickling overhead
# Also set n_jobs=1 for the individual estimators inside the voting ensemble to reduce serialization
voting_estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=1)),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=1)),
    ('ridge', Ridge(alpha=1.0, random_state=42))
]

print(f"\nCreating Voting Regressor with {len(voting_estimators)} models:")
for name, _ in voting_estimators:
    print(f"  - {name}")

# Train voting regressor
print("\nTraining Voting Regressor (using thread-based backend to reduce memory pressure)...")
start_time = time.time()
voting_model = VotingRegressor(estimators=voting_estimators, n_jobs=1)
# Use thread-based backend to avoid loky/cloudpickle serialization of large models
with parallel_backend('threading'):
    voting_model.fit(X_train_scaled, y_train)
    y_pred_voting = voting_model.predict(X_test_scaled)
voting_time = time.time() - start_time

r2_voting = r2_score(y_test, y_pred_voting)
mse_voting = mean_squared_error(y_test, y_pred_voting)
mae_voting = mean_absolute_error(y_test, y_pred_voting)
ev_voting = explained_variance_score(y_test, y_pred_voting)

print(f"\nVoting Regressor Results:")
print(f"  R²: {r2_voting:.4f}")
print(f"  MSE: {mse_voting:.4f}")
print(f"  MAE: {mae_voting:.4f}")
print(f"  Explained Variance: {ev_voting:.4f}")
print(f"  Training Time: {voting_time:.2f}s")

# --- 5. STACKING REGRESSOR (SEQUENTIAL BLENDING) ---
print("\n5. STACKING REGRESSOR - SEQUENTIAL BLENDING WITH META-LEARNER")
print("=" * 70)

print("\nStacking trains base models, then uses their predictions as features")
print("for a meta-learner that learns to optimally combine them.")

# Define base estimators for stacking (Level 0)
stacking_estimators = [
    ('xgb', xgb.XGBRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=150, random_state=42)),
    ('et', ExtraTreesRegressor(n_estimators=150, random_state=42, n_jobs=-1)),
    ('ridge', Ridge(alpha=1.0, random_state=42)),
    ('lasso', Lasso(alpha=0.1, random_state=42))
]

print(f"\nBase models (Level 0) - {len(stacking_estimators)} models:")
for name, _ in stacking_estimators:
    print(f"  - {name}")

# Define meta-learner (Level 1)
meta_learner = Ridge(alpha=1.0, random_state=42)
print(f"\nMeta-learner (Level 1): {type(meta_learner).__name__}")

# Train stacking regressor
print("\nTraining Stacking Regressor...")
start_time = time.time()
stacking_model = StackingRegressor(
    estimators=stacking_estimators,
    final_estimator=meta_learner,
    cv=5,  # 5-fold cross-validation for generating meta-features
    n_jobs=-1
)
stacking_model.fit(X_train_scaled, y_train)
y_pred_stacking = stacking_model.predict(X_test_scaled)
stacking_time = time.time() - start_time

r2_stacking = r2_score(y_test, y_pred_stacking)
mse_stacking = mean_squared_error(y_test, y_pred_stacking)
mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
ev_stacking = explained_variance_score(y_test, y_pred_stacking)

print(f"\nStacking Regressor Results:")
print(f"  R²: {r2_stacking:.4f}")
print(f"  MSE: {mse_stacking:.4f}")
print(f"  MAE: {mae_stacking:.4f}")
print(f"  Explained Variance: {ev_stacking:.4f}")
print(f"  Training Time: {stacking_time:.2f}s")

# Get meta-learner coefficients to see how it weights each base model
print("\nMeta-learner weights for each base model:")
if hasattr(stacking_model.final_estimator_, 'coef_'):
    coef = stacking_model.final_estimator_.coef_
    for i, (name, _) in enumerate(stacking_estimators):
        print(f"  {name}: {coef[i]:.4f}")

# --- 6. COMPARE ALL APPROACHES ---
print("\n6. COMPREHENSIVE COMPARISON")
print("=" * 70)

# Add ensemble results
ensemble_results = individual_results.copy()

ensemble_results.append({
    'model': 'Voting Regressor',
    'ensemble_type': 'Parallel (Voting)',
    'r2_score': r2_voting,
    'mse': mse_voting,
    'mae': mae_voting,
    'explained_variance': ev_voting,
    'training_time_s': voting_time
})

ensemble_results.append({
    'model': 'Stacking Regressor',
    'ensemble_type': 'Sequential (Stacking)',
    'r2_score': r2_stacking,
    'mse': mse_stacking,
    'mae': mae_stacking,
    'explained_variance': ev_stacking,
    'training_time_s': stacking_time
})

# Create final comparison DataFrame
results_df = pd.DataFrame(ensemble_results).sort_values('r2_score', ascending=False)

# Save results
results_csv_path = 'ensemble_results/ensemble_comparison_results.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"\nResults saved to '{results_csv_path}'")

print("\nFinal Rankings (by R² score):")
print(results_df[['model', 'ensemble_type', 'r2_score', 'mse', 'mae', 'training_time_s']].to_string())

# Highlight best performers
best_individual = individual_df.iloc[0]
best_overall = results_df.iloc[0]

print(f"\n{'='*70}")
print("KEY FINDINGS:")
print(f"{'='*70}")
print(f"\nBest Individual Model: {best_individual['model']}")
print(f"  R²: {best_individual['r2_score']:.4f}, MSE: {best_individual['mse']:.4f}")

print(f"\nVoting Regressor (Parallel Ensemble):")
print(f"  R²: {r2_voting:.4f}, MSE: {mse_voting:.4f}")
improvement_voting = ((r2_voting - best_individual['r2_score']) / best_individual['r2_score']) * 100
print(f"  Improvement over best individual: {improvement_voting:+.2f}%")

print(f"\nStacking Regressor (Sequential with Meta-Learner):")
print(f"  R²: {r2_stacking:.4f}, MSE: {mse_stacking:.4f}")
improvement_stacking = ((r2_stacking - best_individual['r2_score']) / best_individual['r2_score']) * 100
print(f"  Improvement over best individual: {improvement_stacking:+.2f}%")

print(f"\nOverall Best Approach: {best_overall['model']} ({best_overall['ensemble_type']})")
print(f"  R²: {best_overall['r2_score']:.4f}, MSE: {best_overall['mse']:.4f}")

# --- 7. VISUALIZATION ---
print("\n7. GENERATING VISUALIZATIONS")
print("=" * 70)

# Create comprehensive visualization
fig = plt.figure(figsize=(24, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. R² Score Comparison
ax1 = fig.add_subplot(gs[0, :])
colors = ['lightblue' if 'Individual' in et else ('green' if 'Voting' in et else 'orange') 
          for et in results_df['ensemble_type']]
bars = ax1.barh(results_df['model'], results_df['r2_score'], color=colors)
ax1.set_xlabel('R² Score', fontsize=12)
ax1.set_title('R² Score Comparison: Individual Models vs Ensembles', fontsize=14, fontweight='bold')
ax1.axvline(x=best_individual['r2_score'], color='red', linestyle='--', label='Best Individual', alpha=0.7)
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# 2. MSE Comparison
ax2 = fig.add_subplot(gs[1, 0])
ensemble_only = results_df[results_df['ensemble_type'] != 'Individual'].copy()
ensemble_only = pd.concat([individual_df.head(3), ensemble_only])
sns.barplot(x='mse', y='model', data=ensemble_only, ax=ax2, orient='h', palette='viridis')
ax2.set_xlabel('MSE', fontsize=11)
ax2.set_title('MSE Comparison\n(Top 3 Individual + Ensembles)', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

# 3. MAE Comparison
ax3 = fig.add_subplot(gs[1, 1])
sns.barplot(x='mae', y='model', data=ensemble_only, ax=ax3, orient='h', palette='plasma')
ax3.set_xlabel('MAE', fontsize=11)
ax3.set_title('MAE Comparison\n(Top 3 Individual + Ensembles)', fontsize=12)
ax3.grid(axis='x', alpha=0.3)

# 4. Training Time Comparison
ax4 = fig.add_subplot(gs[1, 2])
sns.barplot(x='training_time_s', y='model', data=ensemble_only, ax=ax4, orient='h', palette='coolwarm')
ax4.set_xlabel('Training Time (seconds)', fontsize=11)
ax4.set_title('Training Time Comparison', fontsize=12)
ax4.grid(axis='x', alpha=0.3)

# 5. Ensemble Methods Comparison (Side by side)
ax5 = fig.add_subplot(gs[2, 0])
ensemble_comparison = results_df[results_df['ensemble_type'] != 'Individual'][['model', 'r2_score']]
ax5.bar(ensemble_comparison['model'], ensemble_comparison['r2_score'], 
        color=['green', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
ax5.set_ylabel('R² Score', fontsize=11)
ax5.set_title('Ensemble Methods: R² Score', fontsize=12, fontweight='bold')
ax5.set_xticklabels(ensemble_comparison['model'], rotation=15, ha='right')
ax5.grid(axis='y', alpha=0.3)

# 6. Ensemble Methods MSE
ax6 = fig.add_subplot(gs[2, 1])
ensemble_comparison_mse = results_df[results_df['ensemble_type'] != 'Individual'][['model', 'mse']]
ax6.bar(ensemble_comparison_mse['model'], ensemble_comparison_mse['mse'], 
        color=['green', 'orange'], alpha=0.7, edgecolor='black', linewidth=2)
ax6.set_ylabel('MSE', fontsize=11)
ax6.set_title('Ensemble Methods: MSE', fontsize=12, fontweight='bold')
ax6.set_xticklabels(ensemble_comparison_mse['model'], rotation=15, ha='right')
ax6.grid(axis='y', alpha=0.3)

# 7. Improvement over Best Individual
ax7 = fig.add_subplot(gs[2, 2])
improvements = [
    ('Voting\n(Parallel)', improvement_voting),
    ('Stacking\n(Sequential)', improvement_stacking)
]
improvement_names = [x[0] for x in improvements]
improvement_values = [x[1] for x in improvements]
colors_imp = ['green' if v > 0 else 'red' for v in improvement_values]
ax7.bar(improvement_names, improvement_values, color=colors_imp, alpha=0.7, edgecolor='black', linewidth=2)
ax7.set_ylabel('Improvement (%)', fontsize=11)
ax7.set_title('% Improvement over\nBest Individual Model', fontsize=12, fontweight='bold')
ax7.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax7.grid(axis='y', alpha=0.3)

plt.suptitle('ENSEMBLE METHODS ANALYSIS: Voting (Parallel) vs Stacking (Sequential)', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('ensemble_results/ensemble_comparison_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# --- 8. SAVE BEST ENSEMBLE MODEL ---
print("\n8. SAVING BEST ENSEMBLE MODEL FOR PRODUCTION")
print("=" * 70)

# Determine best ensemble model (Voting vs Stacking)
ensemble_models = {
    'Voting': {
        'model': voting_model,
        'r2': r2_voting,
        'mse': mse_voting,
        'mae': mae_voting,
        'type': 'VotingRegressor',
        'description': 'Parallel ensemble averaging predictions from 5 diverse models'
    },
    'Stacking': {
        'model': stacking_model,
        'r2': r2_stacking,
        'mse': mse_stacking,
        'mae': mae_stacking,
        'type': 'StackingRegressor',
        'description': 'Sequential ensemble with 6 base models and Ridge meta-learner'
    }
}

# Find best ensemble by R² score
best_ensemble_name = max(ensemble_models, key=lambda x: ensemble_models[x]['r2'])
best_ensemble_info = ensemble_models[best_ensemble_name]

print(f"\nBest Ensemble Model: {best_ensemble_name}")
print(f"  Type: {best_ensemble_info['type']}")
print(f"  R² Score: {best_ensemble_info['r2']:.4f}")
print(f"  MSE: {best_ensemble_info['mse']:.4f}")
print(f"  MAE: {best_ensemble_info['mae']:.4f}")

# Save the best ensemble model
model_path = 'ensemble_results/models/best_ensemble_model.joblib'
dump(best_ensemble_info['model'], model_path)
print(f"\n✓ Model saved to: {model_path}")

# Create metadata
metadata = {
    'model_name': best_ensemble_name,
    'model_type': best_ensemble_info['type'],
    'description': best_ensemble_info['description'],
    'performance_metrics': {
        'r2_score': float(best_ensemble_info['r2']),
        'mse': float(best_ensemble_info['mse']),
        'mae': float(best_ensemble_info['mae']),
        'explained_variance': float(ev_stacking if best_ensemble_name == 'Stacking' else ev_voting)
    },
    'training_details': {
        'training_time_seconds': float(stacking_time if best_ensemble_name == 'Stacking' else voting_time),
        'n_samples_train': int(len(X_train)),
        'n_samples_test': int(len(X_test)),
        'n_features': int(X_train.shape[1])
    },
    'ensemble_configuration': {
        'n_base_models': len(stacking_estimators if best_ensemble_name == 'Stacking' else voting_estimators),
        'base_models': [name for name, _ in (stacking_estimators if best_ensemble_name == 'Stacking' else voting_estimators)],
        'meta_learner': 'Ridge (alpha=1.0)' if best_ensemble_name == 'Stacking' else 'Simple Average'
    },
    'feature_names': list(X.columns),
    'preprocessing': {
        'scaling': 'StandardScaler',
        'imputation': 'KNNImputer (n_neighbors=5)',
        'dropped_features': ['Apparent Temperature (C)', 'Formatted Date', 'Daily Summary']
    }
}

# Save metadata
metadata_path = 'ensemble_results/models/model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata saved to: {metadata_path}")

print("\n" + "="*70)
print("MODEL PERSISTENCE COMPLETE")
print("="*70)
print("\nSaved files:")
print(f"  1. {model_path}")
print(f"  2. {metadata_path}")
print("\nNote: Scaler is already fitted during data preprocessing.")
print("      Use the same StandardScaler pipeline for new predictions.")

print("\n=== ANALYSIS COMPLETE ===")
print("Generated files:")
print("- ensemble_results/ensemble_comparison_results.csv")
print("- ensemble_results/ensemble_comparison_visualization.png")
print("- ensemble_results/models/best_ensemble_model.joblib")
print("- ensemble_results/models/model_metadata.json")
print("="*70)
