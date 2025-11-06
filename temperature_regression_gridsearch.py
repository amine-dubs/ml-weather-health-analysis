import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import warnings
import os

warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("=== TEMPERATURE REGRESSION PIPELINE (GRIDSEARCHCV) ===\n")

# Create a directory for results if it doesn't exist
if not os.path.exists('gridsearch_results'):
    os.makedirs('gridsearch_results')

# --- 1. LOAD AND PREPROCESS DATA ---
print("1. LOADING AND PREPROCESSING DATA")
print("=" * 50)

# Load the dataset
df = pd.read_csv('Dataset1.csv')
print(f"Original dataset shape: {df.shape}")

# Drop specified columns
columns_to_drop = ['Apparent Temperature (C)', 'Formatted Date', 'Daily Summary']
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")
print(f"Dataset shape after dropping columns: {df.shape}")

# Separate features and target
X = df.drop('Temperature (C)', axis=1)
y = df['Temperature (C)']

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

print(f"\nCategorical features: {list(categorical_features)}")
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
print("\nImputing missing values with KNNImputer...")
imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

print(f"Total missing values after imputation: {X.isnull().sum().sum()}")
print("Preprocessing complete.")

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. HYPERPARAMETER TUNING WITH GRIDSEARCHCV ---
print("\n2. HYPERPARAMETER TUNING WITH GRIDSEARCHCV")
print("=" * 50)

# Define models and parameter grids
# Note: Grids are kept small to reduce runtime. Increase values for more exhaustive search.
models_and_params = {
    'XGBoost': {
        'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.1, 0.2]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }
    },
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [5, 7],
            'learning_rate': [0.1, 0.2]
        }
    }
}

results = []

for model_name, mp in models_and_params.items():
    print(f"\n--- Tuning {model_name} ---")
    start_time = time.time()
    
    grid_search = GridSearchCV(
        estimator=mp['model'],
        param_grid=mp['params'],
        cv=3,  # Using 3-fold cross-validation for speed
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    ev = explained_variance_score(y_test, y_pred)
    elapsed = time.time() - start_time
    
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best R² score from CV: {grid_search.best_score_:.4f}")
    print(f"Test Set R²: {r2:.4f}")
    print(f"Test Set MSE: {mse:.4f}")
    print(f"Tuning and evaluation took {elapsed:.2f}s")
    
    results.append({
        'model': model_name,
        'best_params': str(grid_search.best_params_),
        'r2_score': r2,
        'mse': mse,
        'explained_variance': ev
    })

# --- 3. LOG AND COMPARE RESULTS ---
print("\n3. LOGGING AND COMPARING RESULTS")
print("=" * 50)

# Create DataFrame from GridSearchCV results
gridsearch_results_df = pd.DataFrame(results)
gridsearch_results_df = gridsearch_results_df.sort_values('r2_score', ascending=False)

# Save results to CSV
gridsearch_csv_path = 'gridsearch_results/gridsearch_model_comparison.csv'
gridsearch_results_df.to_csv(gridsearch_csv_path, index=False)
print(f"GridSearchCV results saved to '{gridsearch_csv_path}'")
print("\nGridSearchCV Results Summary:")
print(gridsearch_results_df.to_string())

# Compare with manual optimization results if available
manual_results_path = 'regression_results/model_comparison_results.csv'
if os.path.exists(manual_results_path):
    print("\n--- COMPARISON WITH MANUAL OPTIMIZATION ---")
    manual_results_df = pd.read_csv(manual_results_path)
    
    # Get best result from manual tuning
    best_manual_model = manual_results_df.sort_values('r2_score', ascending=False).iloc[0]
    
    # Get best result from GridSearchCV
    best_gridsearch_model = gridsearch_results_df.sort_values('r2_score', ascending=False).iloc[0]
    
    print("\nBest Manual Tuning Result:")
    print(f"  Model: {best_manual_model['model']}")
    print(f"  R²: {best_manual_model['r2_score']:.4f}, MSE: {best_manual_model['mse']:.4f}")
    print(f"  Settings: n_estimators={best_manual_model['n_estimators']}, data_pct={best_manual_model['data_percentage']}%, scaler={best_manual_model['scaler']}")

    print("\nBest GridSearchCV Result:")
    print(f"  Model: {best_gridsearch_model['model']}")
    print(f"  R²: {best_gridsearch_model['r2_score']:.4f}, MSE: {best_gridsearch_model['mse']:.4f}")
    print(f"  Settings: {best_gridsearch_model['best_params']}")
    
    r2_diff = best_gridsearch_model['r2_score'] - best_manual_model['r2_score']
    print(f"\nDifference in R² (GridSearch - Manual): {r2_diff:+.4f}")
    if r2_diff > 0:
        print("GridSearchCV found a better model configuration.")
    else:
        print("Manual tuning achieved a comparable or better result.")

# Plot final model comparison from GridSearchCV
fig, axes = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Model Performance Comparison (GridSearchCV)', fontsize=16)

# R-squared
sns.barplot(x='r2_score', y='model', data=gridsearch_results_df, ax=axes[0], orient='h')
axes[0].set_title('R-squared (R²)')
axes[0].set_xlabel('R² Score')
axes[0].set_ylabel('Model')

# Mean Squared Error
sns.barplot(x='mse', y='model', data=gridsearch_results_df, ax=axes[1], orient='h')
axes[1].set_title('Mean Squared Error (MSE)')
axes[1].set_xlabel('MSE')
axes[1].set_ylabel('')

# Explained Variance
sns.barplot(x='explained_variance', y='model', data=gridsearch_results_df, ax=axes[2], orient='h')
axes[2].set_title('Explained Variance Score')
axes[2].set_xlabel('Explained Variance')
axes[2].set_ylabel('')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('gridsearch_results/gridsearch_final_model_comparison.png', dpi=300)
plt.show()

print("\n=== ANALYSIS COMPLETE ===")
print("Generated files:")
print("- gridsearch_results/gridsearch_model_comparison.csv")
print("- gridsearch_results/gridsearch_final_model_comparison.png")
