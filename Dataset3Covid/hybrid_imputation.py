"""
Hybrid Imputation Strategy - Combining KNN and MICE
Uses the best imputation method for each feature based on performance
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Load data
print("="*70)
print(" HYBRID IMPUTATION OPTIMIZATION")
print("="*70)

print("\nLoading dataset...")
df = pd.read_csv(r'C:\Users\LENOVO\Desktop\ML tp0\Dataset3Covid\Dataset3.csv')
print(f"Dataset: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# Separate target
target = df['SARSCov'].copy()
features = df.drop('SARSCov', axis=1)
feature_names = features.columns.tolist()

# Store original statistics
original_stats = {}
for col in features.columns:
    if features[col].isnull().sum() > 0:
        original_stats[col] = {
            'mean': features[col].mean(),
            'std': features[col].std(),
            'missing_pct': (features[col].isnull().sum() / len(features)) * 100
        }

print(f"\nFeatures with missing values: {len(original_stats)}")

# ============================================================================
# STEP 1: Test KNN k=5 on each feature individually
# ============================================================================
print("\n" + "="*70)
print("STEP 1: Testing KNN (k=5) on each feature")
print("="*70)

knn_changes = {}
knn_imputer = KNNImputer(n_neighbors=20)
features_knn = pd.DataFrame(
    knn_imputer.fit_transform(features),
    columns=features.columns
)

for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    knn_mean = features_knn[col].mean()
    mean_change = abs((knn_mean - orig_mean) / orig_mean) * 100
    knn_changes[col] = mean_change

# Sort features by KNN performance
knn_sorted = sorted(knn_changes.items(), key=lambda x: x[1])
print(f"\nTop 10 features with lowest KNN change:")
print(f"{'Feature':<10} {'Mean Change %':>15} {'Missing %':>12}")
print("-"*70)
for feat, change in knn_sorted[:10]:
    print(f"{feat:<10} {change:>14.2f}% {original_stats[feat]['missing_pct']:>11.1f}%")

# ============================================================================
# STEP 2: Test MICE on each feature individually
# ============================================================================
print("\n" + "="*70)
print("STEP 2: Testing MICE (iter=10) on each feature")
print("="*70)

mice_changes = {}
mice_imputer = IterativeImputer(max_iter=10, random_state=42)
features_mice = pd.DataFrame(
    mice_imputer.fit_transform(features),
    columns=features.columns
)

for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    mice_mean = features_mice[col].mean()
    mean_change = abs((mice_mean - orig_mean) / orig_mean) * 100
    mice_changes[col] = mean_change

# Sort features by MICE performance
mice_sorted = sorted(mice_changes.items(), key=lambda x: x[1])
print(f"\nTop 10 features with lowest MICE change:")
print(f"{'Feature':<10} {'Mean Change %':>15} {'Missing %':>12}")
print("-"*70)
for feat, change in mice_sorted[:10]:
    print(f"{feat:<10} {change:>14.2f}% {original_stats[feat]['missing_pct']:>11.1f}%")

# ============================================================================
# STEP 3: Create hybrid strategy - choose best method per feature
# ============================================================================
print("\n" + "="*70)
print("STEP 3: Creating Hybrid Strategy (Best of Both)")
print("="*70)

hybrid_assignment = {}
knn_features = []
mice_features = []

print(f"\n{'Feature':<10} {'KNN %':>10} {'MICE %':>10} {'Best':>10} {'Missing %':>12}")
print("-"*70)

for col in original_stats.keys():
    knn_change = knn_changes[col]
    mice_change = mice_changes[col]
    missing_pct = original_stats[col]['missing_pct']
    
    # Choose the method with lower change
    if knn_change <= mice_change:
        hybrid_assignment[col] = 'KNN'
        knn_features.append(col)
        best_change = knn_change
    else:
        hybrid_assignment[col] = 'MICE'
        mice_features.append(col)
        best_change = mice_change
    
    print(f"{col:<10} {knn_change:>9.2f}% {mice_change:>9.2f}% {hybrid_assignment[col]:>10} {missing_pct:>11.1f}%")

print(f"\n{'='*70}")
print(f"Features assigned to KNN: {len(knn_features)}")
print(f"Features assigned to MICE: {len(mice_features)}")

# ============================================================================
# STEP 4: Apply hybrid imputation
# ============================================================================
print("\n" + "="*70)
print("STEP 4: Applying Hybrid Imputation")
print("="*70)

# Start with original features
features_hybrid = features.copy()

# If we have features for KNN
if knn_features:
    print(f"\nApplying KNN (k=5) to {len(knn_features)} features...")
    # Get all features for KNN imputation (it needs all features)
    knn_result = pd.DataFrame(
        knn_imputer.fit_transform(features),
        columns=features.columns
    )
    # Only keep the columns assigned to KNN
    for col in knn_features:
        features_hybrid[col] = knn_result[col]
    print(f"  KNN features: {', '.join(knn_features)}")

# If we have features for MICE
if mice_features:
    print(f"\nApplying MICE (iter=10) to {len(mice_features)} features...")
    # Get all features for MICE imputation (it needs all features)
    mice_result = pd.DataFrame(
        mice_imputer.fit_transform(features),
        columns=features.columns
    )
    # Only keep the columns assigned to MICE
    for col in mice_features:
        features_hybrid[col] = mice_result[col]
    print(f"  MICE features: {', '.join(mice_features)}")

# ============================================================================
# STEP 5: Evaluate hybrid results
# ============================================================================
print("\n" + "="*70)
print("STEP 5: Evaluating Hybrid Performance")
print("="*70)

hybrid_changes = {}
total_mean_change = 0
total_std_change = 0
max_mean_change = 0
max_mean_feature = ""

print(f"\n{'Feature':<10} {'Method':>8} {'Mean Δ %':>12} {'Std Δ %':>12} {'Missing %':>12}")
print("-"*70)

for col in original_stats.keys():
    orig_mean = original_stats[col]['mean']
    orig_std = original_stats[col]['std']
    hybrid_mean = features_hybrid[col].mean()
    hybrid_std = features_hybrid[col].std()
    
    mean_change = abs((hybrid_mean - orig_mean) / orig_mean) * 100
    std_change = abs((hybrid_std - orig_std) / orig_std) * 100
    
    hybrid_changes[col] = mean_change
    total_mean_change += mean_change
    total_std_change += std_change
    
    if mean_change > max_mean_change:
        max_mean_change = mean_change
        max_mean_feature = col
    
    print(f"{col:<10} {hybrid_assignment[col]:>8} {mean_change:>11.2f}% {std_change:>11.2f}% {original_stats[col]['missing_pct']:>11.1f}%")

avg_mean_change = total_mean_change / len(original_stats)
avg_std_change = total_std_change / len(original_stats)

# ============================================================================
# STEP 6: Compare all methods
# ============================================================================
print("\n" + "="*70)
print("FINAL COMPARISON - All Methods")
print("="*70)

# Calculate averages for each method
knn_avg = sum(knn_changes.values()) / len(knn_changes)
mice_avg = sum(mice_changes.values()) / len(mice_changes)
hybrid_avg = avg_mean_change

print(f"\n{'Method':<25} {'Avg Mean Change':>20} {'Status':>15}")
print("-"*70)
print(f"{'KNN (k=5)':<25} {knn_avg:>18.2f}% {'':<15}")
print(f"{'MICE (iter=10)':<25} {mice_avg:>18.2f}% {'':<15}")
print(f"{'Hybrid (KNN + MICE)':<25} {hybrid_avg:>18.2f}% {'✓✓ BEST!' if hybrid_avg < min(knn_avg, mice_avg) else '✓ Good':>15}")

improvement_vs_knn = ((knn_avg - hybrid_avg) / knn_avg) * 100
improvement_vs_mice = ((mice_avg - hybrid_avg) / mice_avg) * 100

print(f"\n{'='*70}")
print(f"HYBRID RESULTS:")
print(f"  Average Mean Change: {hybrid_avg:.2f}%")
print(f"  Average Std Change: {avg_std_change:.2f}%")
print(f"  Max Mean Change: {max_mean_change:.2f}% ({max_mean_feature})")
print(f"  Improvement vs KNN: {improvement_vs_knn:.1f}%")
print(f"  Improvement vs MICE: {improvement_vs_mice:.1f}%")
print(f"  Target (<2%): {'✓✓ ACHIEVED!' if hybrid_avg < 2.0 else '✗ Not achieved'}")
print(f"{'='*70}")

# ============================================================================
# STEP 7: Save results and configuration
# ============================================================================
print("\n" + "="*70)
print("STEP 7: Saving Results")
print("="*70)

# Save comparison results
comparison_df = pd.DataFrame({
    'Feature': list(original_stats.keys()),
    'Missing_Pct': [original_stats[f]['missing_pct'] for f in original_stats.keys()],
    'KNN_Change_Pct': [knn_changes[f] for f in original_stats.keys()],
    'MICE_Change_Pct': [mice_changes[f] for f in original_stats.keys()],
    'Hybrid_Change_Pct': [hybrid_changes[f] for f in original_stats.keys()],
    'Hybrid_Method': [hybrid_assignment[f] for f in original_stats.keys()]
}).sort_values('Hybrid_Change_Pct')

comparison_df.to_csv('covid_results/hybrid_imputation_comparison.csv', index=False)
print("✓ Comparison saved: covid_results/hybrid_imputation_comparison.csv")

# Save hybrid configuration
with open('covid_results/hybrid_imputation_config.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("HYBRID IMPUTATION CONFIGURATION\n")
    f.write("="*70 + "\n\n")
    f.write(f"Average Mean Change: {hybrid_avg:.2f}%\n")
    f.write(f"Average Std Change: {avg_std_change:.2f}%\n")
    f.write(f"Max Mean Change: {max_mean_change:.2f}% ({max_mean_feature})\n\n")
    f.write(f"KNN Features ({len(knn_features)}):\n")
    for feat in sorted(knn_features):
        f.write(f"  - {feat}: {knn_changes[feat]:.2f}% change, {original_stats[feat]['missing_pct']:.1f}% missing\n")
    f.write(f"\nMICE Features ({len(mice_features)}):\n")
    for feat in sorted(mice_features):
        f.write(f"  - {feat}: {mice_changes[feat]:.2f}% change, {original_stats[feat]['missing_pct']:.1f}% missing\n")
    f.write("\n" + "="*70 + "\n")
    f.write(f"Improvement vs KNN only: {improvement_vs_knn:.1f}%\n")
    f.write(f"Improvement vs MICE only: {improvement_vs_mice:.1f}%\n")
    f.write("="*70 + "\n")

print("✓ Configuration saved: covid_results/hybrid_imputation_config.txt")

# Save the hybrid imputed dataset for future use
features_hybrid['SARSCov'] = target
features_hybrid.to_csv('covid_results/dataset_hybrid_imputed.csv', index=False)
print("✓ Hybrid imputed dataset saved: covid_results/dataset_hybrid_imputed.csv")

print("\n" + "="*70)
print("HYBRID IMPUTATION OPTIMIZATION COMPLETE!")
print("="*70)
