"""
Optimize Imputation Strategy to Minimize Distribution Changes
Goal: Reduce average mean change to < 2%
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" IMPUTATION OPTIMIZATION - Finding Best Strategy")
print("="*70)

# Load data
print("\nLoading dataset...")
df = pd.read_csv(r'C:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\Dataset3.csv')
target = df['SARSCov'].copy()
features_orig = df.drop('SARSCov', axis=1)

print(f"Dataset: {df.shape}")
print(f"Missing values: {features_orig.isnull().sum().sum()}")

def evaluate_imputation(imputer, name):
    """Evaluate imputation quality"""
    # Get features with missing values
    missing_mask = features_orig.isnull().any()
    features_with_missing = features_orig.columns[missing_mask].tolist()
    
    # Apply imputation
    features_imputed = pd.DataFrame(
        imputer.fit_transform(features_orig),
        columns=features_orig.columns
    )
    
    # Calculate statistics
    results = []
    for col in features_with_missing:
        orig_data = features_orig[col].dropna()
        imp_data = features_imputed[col]
        
        orig_mean = orig_data.mean()
        imp_mean = imp_data.mean()
        mean_change_pct = abs(imp_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
        
        orig_std = orig_data.std()
        imp_std = imp_data.std()
        std_change_pct = abs(imp_std - orig_std) / orig_std * 100 if orig_std != 0 else 0
        
        results.append({
            'feature': col,
            'mean_change_pct': mean_change_pct,
            'std_change_pct': std_change_pct
        })
    
    # Calculate average
    avg_mean_change = np.mean([r['mean_change_pct'] for r in results])
    avg_std_change = np.mean([r['std_change_pct'] for r in results])
    max_mean_change = np.max([r['mean_change_pct'] for r in results])
    
    return {
        'name': name,
        'avg_mean_change': avg_mean_change,
        'avg_std_change': avg_std_change,
        'max_mean_change': max_mean_change,
        'details': results
    }

print("\n" + "="*70)
print(" TESTING DIFFERENT IMPUTATION STRATEGIES")
print("="*70)

strategies = []

# 1. KNN with different k values
print("\n1. Testing KNN Imputation with different k values...")
for k in [3, 5, 7, 10, 15, 20]:
    print(f"   - Testing k={k}...", end=" ")
    imputer = KNNImputer(n_neighbors=k)
    result = evaluate_imputation(imputer, f'KNN (k={k})')
    strategies.append(result)
    print(f"Avg change: {result['avg_mean_change']:.2f}%")

# 2. Iterative Imputation (MICE) with different estimators
print("\n2. Testing Iterative Imputation (MICE)...")

# MICE with different max_iter
for max_iter in [10, 20, 50]:
    print(f"   - Testing MICE (max_iter={max_iter})...", end=" ")
    imputer = IterativeImputer(max_iter=max_iter, random_state=42, verbose=0)
    result = evaluate_imputation(imputer, f'MICE (iter={max_iter})')
    strategies.append(result)
    print(f"Avg change: {result['avg_mean_change']:.2f}%")

# MICE with Random Forest
print(f"   - Testing MICE with Random Forest...", end=" ")
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42,
    verbose=0
)
result = evaluate_imputation(imputer, 'MICE (RF)')
strategies.append(result)
print(f"Avg change: {result['avg_mean_change']:.2f}%")

# 3. Hybrid approach - median for high missing, KNN for low missing
print("\n3. Testing Hybrid Approach...")
print(f"   - Median for >30% missing, KNN for <=30% missing...", end=" ")

# Calculate missing percentages
missing_pct = features_orig.isnull().sum() / len(features_orig) * 100

# High missing features (>30%)
high_missing_cols = missing_pct[missing_pct > 30].index.tolist()
low_missing_cols = [col for col in features_orig.columns if col not in high_missing_cols]

# Apply hybrid imputation
features_hybrid = features_orig.copy()

# Median for high missing
if high_missing_cols:
    median_imputer = SimpleImputer(strategy='median')
    features_hybrid[high_missing_cols] = median_imputer.fit_transform(features_hybrid[high_missing_cols])

# KNN for low missing
if low_missing_cols:
    knn_imputer = KNNImputer(n_neighbors=5)
    features_hybrid[low_missing_cols] = knn_imputer.fit_transform(features_hybrid[low_missing_cols])

# Evaluate hybrid
features_with_missing = features_orig.columns[features_orig.isnull().any()].tolist()
results = []
for col in features_with_missing:
    orig_data = features_orig[col].dropna()
    imp_data = features_hybrid[col]
    
    orig_mean = orig_data.mean()
    imp_mean = imp_data.mean()
    mean_change_pct = abs(imp_mean - orig_mean) / orig_mean * 100 if orig_mean != 0 else 0
    
    orig_std = orig_data.std()
    imp_std = imp_data.std()
    std_change_pct = abs(imp_std - orig_std) / orig_std * 100 if orig_std != 0 else 0
    
    results.append({
        'feature': col,
        'mean_change_pct': mean_change_pct,
        'std_change_pct': std_change_pct
    })

hybrid_result = {
    'name': 'Hybrid (Median+KNN)',
    'avg_mean_change': np.mean([r['mean_change_pct'] for r in results]),
    'avg_std_change': np.mean([r['std_change_pct'] for r in results]),
    'max_mean_change': np.max([r['mean_change_pct'] for r in results]),
    'details': results
}
strategies.append(hybrid_result)
print(f"Avg change: {hybrid_result['avg_mean_change']:.2f}%")

# Sort by average mean change
strategies.sort(key=lambda x: x['avg_mean_change'])

print("\n" + "="*70)
print(" RESULTS COMPARISON")
print("="*70)

print(f"\n{'Rank':<6} {'Strategy':<25} {'Avg Mean Δ':<15} {'Avg Std Δ':<15} {'Max Mean Δ':<15}")
print("-"*70)

for i, s in enumerate(strategies, 1):
    status = "✓✓ BEST!" if i == 1 else ("✓ Good" if s['avg_mean_change'] < 2.0 else "")
    print(f"{i:<6} {s['name']:<25} {s['avg_mean_change']:>12.2f}%  {s['avg_std_change']:>12.2f}%  {s['max_mean_change']:>12.2f}%  {status}")

# Get best strategy
best = strategies[0]

print("\n" + "="*70)
print(" BEST IMPUTATION STRATEGY")
print("="*70)
print(f"\nStrategy: {best['name']}")
print(f"Average Mean Change: {best['avg_mean_change']:.2f}%")
print(f"Average Std Change: {best['avg_std_change']:.2f}%")
print(f"Max Mean Change: {best['max_mean_change']:.2f}%")

if best['avg_mean_change'] < 2.0:
    print(f"\n✓✓ SUCCESS! Achieved < 2% average mean change!")
else:
    print(f"\n⚠ Close but not quite. Current best is {best['avg_mean_change']:.2f}%")

# Show top features with largest changes
print("\n" + "="*70)
print(" FEATURES WITH LARGEST CHANGES (Best Strategy)")
print("="*70)

best_details = sorted(best['details'], key=lambda x: x['mean_change_pct'], reverse=True)
print(f"\n{'Rank':<6} {'Feature':<10} {'Mean Change %':<15} {'Std Change %':<15}")
print("-"*70)
for i, detail in enumerate(best_details[:10], 1):
    print(f"{i:<6} {detail['feature']:<10} {detail['mean_change_pct']:>12.2f}%  {detail['std_change_pct']:>12.2f}%")

# Save best imputer configuration
print("\n" + "="*70)
print(" SAVING BEST CONFIGURATION")
print("="*70)

# Determine which imputer to use based on best strategy
if 'KNN' in best['name']:
    k_value = int(best['name'].split('k=')[1].rstrip(')'))
    best_imputer = KNNImputer(n_neighbors=k_value)
    config = f"KNN with k={k_value}"
elif 'MICE' in best['name']:
    if 'RF' in best['name']:
        best_imputer = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, random_state=42),
            max_iter=10,
            random_state=42,
            verbose=0
        )
        config = "MICE with Random Forest estimator"
    else:
        max_iter = int(best['name'].split('iter=')[1].rstrip(')'))
        best_imputer = IterativeImputer(max_iter=max_iter, random_state=42, verbose=0)
        config = f"MICE with max_iter={max_iter}"
else:
    config = "Hybrid (Median for >30% missing, KNN k=5 for others)"

# Save recommendation
with open('covid_results/best_imputation_config.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write(" OPTIMIZED IMPUTATION CONFIGURATION\n")
    f.write("="*70 + "\n\n")
    f.write(f"Best Strategy: {best['name']}\n")
    f.write(f"Configuration: {config}\n\n")
    f.write(f"Performance Metrics:\n")
    f.write(f"  - Average Mean Change: {best['avg_mean_change']:.2f}%\n")
    f.write(f"  - Average Std Change: {best['avg_std_change']:.2f}%\n")
    f.write(f"  - Maximum Mean Change: {best['max_mean_change']:.2f}%\n\n")
    f.write(f"Goal Achievement: {'YES - < 2%' if best['avg_mean_change'] < 2.0 else 'NO - Close but above 2%'}\n\n")
    f.write("Features with largest changes:\n")
    for i, detail in enumerate(best_details[:10], 1):
        f.write(f"  {i}. {detail['feature']}: {detail['mean_change_pct']:.2f}% mean change\n")
    f.write("\n" + "="*70 + "\n")
    f.write("All Strategies Tested (sorted by performance):\n")
    f.write("="*70 + "\n")
    for i, s in enumerate(strategies, 1):
        f.write(f"{i}. {s['name']}: {s['avg_mean_change']:.2f}% avg mean change\n")

print(f"\n✓ Configuration saved: covid_results/best_imputation_config.txt")

# Save detailed results
results_df = pd.DataFrame([
    {
        'Strategy': s['name'],
        'Avg_Mean_Change_%': s['avg_mean_change'],
        'Avg_Std_Change_%': s['avg_std_change'],
        'Max_Mean_Change_%': s['max_mean_change']
    }
    for s in strategies
])
results_df.to_csv('covid_results/imputation_optimization_results.csv', index=False)
print(f"✓ Results saved: covid_results/imputation_optimization_results.csv")

print("\n" + "="*70)
print(" OPTIMIZATION COMPLETE")
print("="*70)
print(f"\nBest Strategy: {best['name']}")
print(f"Average Mean Change: {best['avg_mean_change']:.2f}%")
print(f"Target: < 2.00%")
print(f"Status: {'✓✓ ACHIEVED!' if best['avg_mean_change'] < 2.0 else f'Close - Only {best['avg_mean_change'] - 2.0:.2f}% above target'}")
print("="*70)
