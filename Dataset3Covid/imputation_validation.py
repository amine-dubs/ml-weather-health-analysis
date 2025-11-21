"""
Imputation Validation Analysis
Compares data distributions before and after KNN imputation
Verifies that imputation preserves statistical properties
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.impute import KNNImputer
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print(" IMPUTATION VALIDATION ANALYSIS")
print("="*80)

# Setup directories
output_dir = 'covid_results'
plots_dir = os.path.join(output_dir, 'plots')
os.makedirs(plots_dir, exist_ok=True)

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("\n1. LOADING DATA")
print("="*80)

# Get dataset path
script_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, 'Dataset3.csv')

if not os.path.exists(dataset_path):
    dataset_path = os.path.join(os.path.dirname(script_dir), 'Dataset3Covid', 'Dataset3.csv')

print(f"Loading dataset from: {dataset_path}")
df_original = pd.read_csv(dataset_path)
print(f"✓ Dataset loaded: {df_original.shape}")

# Separate features and target
X_original = df_original.drop('SARSCov', axis=1)
y = df_original['SARSCov']

print(f"\nOriginal data:")
print(f"  Features: {X_original.shape[1]}")
print(f"  Samples: {len(X_original)}")
print(f"  Missing values: {X_original.isnull().sum().sum()}")
print(f"  Missing percentage: {X_original.isnull().sum().sum() / X_original.size * 100:.2f}%")

# ============================================================================
# 2. IMPUTATION
# ============================================================================

print("\n2. PERFORMING KNN IMPUTATION")
print("="*80)

# Apply KNN Imputation
imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed_array = imputer.fit_transform(X_original)
X_imputed = pd.DataFrame(X_imputed_array, columns=X_original.columns, index=X_original.index)

print(f"✓ Imputation complete")
print(f"  Missing values after: {X_imputed.isnull().sum().sum()}")

# ============================================================================
# 3. STATISTICAL COMPARISON
# ============================================================================

print("\n3. STATISTICAL COMPARISON (BEFORE vs AFTER)")
print("="*80)

comparison_results = []

for col in X_original.columns:
    # Original data (non-missing only)
    original_non_missing = X_original[col].dropna()
    
    # Imputed data (all values)
    imputed_all = X_imputed[col]
    
    # Imputed values only (where original was NaN)
    imputed_only = X_imputed[col][X_original[col].isnull()]
    
    # Calculate statistics
    result = {
        'Feature': col,
        'Missing_Count': X_original[col].isnull().sum(),
        'Missing_Pct': X_original[col].isnull().sum() / len(X_original) * 100,
        
        # Original (non-missing) statistics
        'Original_Mean': original_non_missing.mean(),
        'Original_Median': original_non_missing.median(),
        'Original_Std': original_non_missing.std(),
        'Original_Min': original_non_missing.min(),
        'Original_Max': original_non_missing.max(),
        
        # After imputation (all values)
        'Imputed_Mean': imputed_all.mean(),
        'Imputed_Median': imputed_all.median(),
        'Imputed_Std': imputed_all.std(),
        'Imputed_Min': imputed_all.min(),
        'Imputed_Max': imputed_all.max(),
        
        # Imputed values only
        'ImputedVals_Mean': imputed_only.mean() if len(imputed_only) > 0 else np.nan,
        'ImputedVals_Median': imputed_only.median() if len(imputed_only) > 0 else np.nan,
        'ImputedVals_Std': imputed_only.std() if len(imputed_only) > 0 else np.nan,
        
        # Changes
        'Mean_Change': imputed_all.mean() - original_non_missing.mean(),
        'Mean_Change_Pct': ((imputed_all.mean() - original_non_missing.mean()) / original_non_missing.mean() * 100) if original_non_missing.mean() != 0 else 0,
        'Std_Change': imputed_all.std() - original_non_missing.std(),
        'Std_Change_Pct': ((imputed_all.std() - original_non_missing.std()) / original_non_missing.std() * 100) if original_non_missing.std() != 0 else 0
    }
    
    comparison_results.append(result)

# Create comparison DataFrame
comparison_df = pd.DataFrame(comparison_results)
comparison_df = comparison_df.sort_values('Missing_Pct', ascending=False)

# Display features with missing values
features_with_missing = comparison_df[comparison_df['Missing_Count'] > 0]

print(f"\nFeatures with missing values: {len(features_with_missing)}")
print("\nTop 10 Features by Missing Percentage:")
print("="*80)
display_cols = ['Feature', 'Missing_Count', 'Missing_Pct', 'Original_Mean', 'Imputed_Mean', 'Mean_Change_Pct']
print(features_with_missing[display_cols].head(10).to_string(index=False))

print("\n\nDetailed Statistics Comparison (Top 10 by Missing %):")
print("="*80)
for idx, row in features_with_missing.head(10).iterrows():
    print(f"\n{row['Feature']}:")
    print(f"  Missing: {row['Missing_Count']} ({row['Missing_Pct']:.2f}%)")
    print(f"  Original  - Mean: {row['Original_Mean']:.4f}, Std: {row['Original_Std']:.4f}, Median: {row['Original_Median']:.4f}")
    print(f"  Imputed   - Mean: {row['Imputed_Mean']:.4f}, Std: {row['Imputed_Std']:.4f}, Median: {row['Imputed_Median']:.4f}")
    print(f"  Change    - Mean: {row['Mean_Change']:+.4f} ({row['Mean_Change_Pct']:+.2f}%), Std: {row['Std_Change']:+.4f} ({row['Std_Change_Pct']:+.2f}%)")

# ============================================================================
# 4. DISTRIBUTION PRESERVATION ANALYSIS
# ============================================================================

print("\n4. DISTRIBUTION PRESERVATION ANALYSIS")
print("="*80)

# Kolmogorov-Smirnov test for distribution similarity
ks_results = []

for col in X_original.columns:
    if X_original[col].isnull().sum() > 0:
        original_non_missing = X_original[col].dropna()
        imputed_non_missing = X_imputed[col][X_original[col].notna()]
        
        # KS test (comparing original non-missing with same values in imputed dataset)
        ks_stat, ks_pval = stats.ks_2samp(original_non_missing, imputed_non_missing)
        
        ks_results.append({
            'Feature': col,
            'KS_Statistic': ks_stat,
            'KS_PValue': ks_pval,
            'Distribution_Preserved': 'Yes' if ks_pval > 0.05 else 'No'
        })

ks_df = pd.DataFrame(ks_results)
print("\nKolmogorov-Smirnov Test (Distribution Similarity):")
print("(Tests if imputation preserved original distribution)")
print("-"*80)
print(ks_df.to_string(index=False))

preserved_count = (ks_df['Distribution_Preserved'] == 'Yes').sum()
print(f"\n✓ Distributions preserved: {preserved_count}/{len(ks_df)} features ({preserved_count/len(ks_df)*100:.1f}%)")

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

print("\n5. GENERATING VISUALIZATIONS")
print("="*80)

# 5.1 Mean comparison plot
plt.figure(figsize=(12, 8))
features_plot = features_with_missing.head(15)
x = np.arange(len(features_plot))
width = 0.35

bars1 = plt.bar(x - width/2, features_plot['Original_Mean'], width, 
                label='Before Imputation', color='skyblue', alpha=0.8, edgecolor='black')
bars2 = plt.bar(x + width/2, features_plot['Imputed_Mean'], width,
                label='After Imputation', color='coral', alpha=0.8, edgecolor='black')

plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Mean Value', fontsize=12, fontweight='bold')
plt.title('Mean Values: Before vs After Imputation (Top 15 by Missing %)', fontsize=14, fontweight='bold')
plt.xticks(x, features_plot['Feature'], rotation=45, ha='right')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'imputation_mean_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: imputation_mean_comparison.png")

# 5.2 Percentage change in mean
plt.figure(figsize=(12, 8))
colors = ['green' if abs(x) < 5 else 'orange' if abs(x) < 10 else 'red' 
          for x in features_with_missing['Mean_Change_Pct']]
bars = plt.barh(range(len(features_with_missing)), features_with_missing['Mean_Change_Pct'], 
                color=colors, alpha=0.7, edgecolor='black')
plt.yticks(range(len(features_with_missing)), features_with_missing['Feature'])
plt.xlabel('Mean Change (%)', fontsize=12, fontweight='bold')
plt.title('Percentage Change in Mean After Imputation', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.axvline(x=-5, color='orange', linestyle=':', alpha=0.5, label='±5% threshold')
plt.axvline(x=5, color='orange', linestyle=':', alpha=0.5)
plt.gca().invert_yaxis()
plt.legend()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'imputation_mean_change_pct.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: imputation_mean_change_pct.png")

# 5.3 Distribution comparison for top features with missing values
features_to_plot = features_with_missing.head(9)['Feature'].tolist()
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.ravel()

for i, col in enumerate(features_to_plot):
    if i >= 9:
        break
    
    # Original non-missing values
    original_vals = X_original[col].dropna()
    # All imputed values
    imputed_vals = X_imputed[col]
    # Only the imputed values (where original was NaN)
    only_imputed = X_imputed[col][X_original[col].isnull()]
    
    # Plot distributions
    axes[i].hist(original_vals, bins=30, alpha=0.5, label='Original', color='blue', density=True)
    axes[i].hist(imputed_vals, bins=30, alpha=0.5, label='After Imputation', color='red', density=True)
    
    if len(only_imputed) > 0:
        axes[i].axvline(only_imputed.mean(), color='green', linestyle='--', 
                       linewidth=2, label='Imputed Mean')
    
    axes[i].set_title(f'{col}\n({X_original[col].isnull().sum()} missing, {X_original[col].isnull().sum()/len(X_original)*100:.1f}%)', 
                     fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'imputation_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: imputation_distributions.png")

# 5.4 Missing values heatmap
plt.figure(figsize=(14, 10))
missing_data = X_original[features_with_missing['Feature'].tolist()].isnull().astype(int)
plt.imshow(missing_data.T, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')
plt.colorbar(label='Missing (1) / Present (0)', ticks=[0, 1])
plt.xlabel('Sample Index', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Missing Values Pattern (Before Imputation)', fontsize=14, fontweight='bold')
plt.yticks(range(len(features_with_missing)), features_with_missing['Feature'], fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'imputation_missing_pattern.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: imputation_missing_pattern.png")

# 5.5 Statistical summary comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Mean comparison scatter
axes[0, 0].scatter(features_with_missing['Original_Mean'], features_with_missing['Imputed_Mean'], 
                   s=100, alpha=0.6, edgecolors='black')
axes[0, 0].plot([features_with_missing['Original_Mean'].min(), features_with_missing['Original_Mean'].max()],
                [features_with_missing['Original_Mean'].min(), features_with_missing['Original_Mean'].max()],
                'r--', linewidth=2, label='Perfect Match')
axes[0, 0].set_xlabel('Original Mean', fontweight='bold')
axes[0, 0].set_ylabel('Imputed Mean', fontweight='bold')
axes[0, 0].set_title('Mean Values Comparison', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Std comparison scatter
axes[0, 1].scatter(features_with_missing['Original_Std'], features_with_missing['Imputed_Std'],
                   s=100, alpha=0.6, edgecolors='black', color='orange')
axes[0, 1].plot([features_with_missing['Original_Std'].min(), features_with_missing['Original_Std'].max()],
                [features_with_missing['Original_Std'].min(), features_with_missing['Original_Std'].max()],
                'r--', linewidth=2, label='Perfect Match')
axes[0, 1].set_xlabel('Original Std', fontweight='bold')
axes[0, 1].set_ylabel('Imputed Std', fontweight='bold')
axes[0, 1].set_title('Standard Deviation Comparison', fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)

# Median comparison scatter
axes[1, 0].scatter(features_with_missing['Original_Median'], features_with_missing['Imputed_Median'],
                   s=100, alpha=0.6, edgecolors='black', color='green')
axes[1, 0].plot([features_with_missing['Original_Median'].min(), features_with_missing['Original_Median'].max()],
                [features_with_missing['Original_Median'].min(), features_with_missing['Original_Median'].max()],
                'r--', linewidth=2, label='Perfect Match')
axes[1, 0].set_xlabel('Original Median', fontweight='bold')
axes[1, 0].set_ylabel('Imputed Median', fontweight='bold')
axes[1, 0].set_title('Median Values Comparison', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(alpha=0.3)

# Change percentage by missing percentage
axes[1, 1].scatter(features_with_missing['Missing_Pct'], features_with_missing['Mean_Change_Pct'],
                   s=features_with_missing['Missing_Count'], alpha=0.6, edgecolors='black', color='purple')
axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=2)
axes[1, 1].axhline(y=5, color='orange', linestyle=':', alpha=0.5)
axes[1, 1].axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
axes[1, 1].set_xlabel('Missing Data %', fontweight='bold')
axes[1, 1].set_ylabel('Mean Change %', fontweight='bold')
axes[1, 1].set_title('Impact of Missing % on Mean Change', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'imputation_statistical_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: imputation_statistical_comparison.png")

print(f"\n✓ All visualizations saved to: {plots_dir}/")

# ============================================================================
# 6. SAVE COMPARISON RESULTS
# ============================================================================

print("\n6. SAVING COMPARISON RESULTS")
print("="*80)

# Save detailed comparison
comparison_path = os.path.join(output_dir, 'imputation_comparison.csv')
comparison_df.to_csv(comparison_path, index=False)
print(f"✓ Detailed comparison saved: {comparison_path}")

# Save only features with missing values
missing_comparison_path = os.path.join(output_dir, 'imputation_missing_features.csv')
features_with_missing.to_csv(missing_comparison_path, index=False)
print(f"✓ Missing features comparison saved: {missing_comparison_path}")

# Save KS test results
ks_path = os.path.join(output_dir, 'imputation_ks_test.csv')
ks_df.to_csv(ks_path, index=False)
print(f"✓ KS test results saved: {ks_path}")

# ============================================================================
# 7. QUALITY ASSESSMENT
# ============================================================================

print("\n7. IMPUTATION QUALITY ASSESSMENT")
print("="*80)

# Calculate quality metrics
small_change = (features_with_missing['Mean_Change_Pct'].abs() < 5).sum()
moderate_change = ((features_with_missing['Mean_Change_Pct'].abs() >= 5) & 
                   (features_with_missing['Mean_Change_Pct'].abs() < 10)).sum()
large_change = (features_with_missing['Mean_Change_Pct'].abs() >= 10).sum()

print(f"\nMean Change Assessment:")
print(f"  ✓ Small change (<5%): {small_change} features ({small_change/len(features_with_missing)*100:.1f}%)")
print(f"  ⚠ Moderate change (5-10%): {moderate_change} features ({moderate_change/len(features_with_missing)*100:.1f}%)")
print(f"  ⚠ Large change (>10%): {large_change} features ({large_change/len(features_with_missing)*100:.1f}%)")

# Features with large changes
if large_change > 0:
    print(f"\nFeatures with large mean changes (>10%):")
    large_change_features = features_with_missing[features_with_missing['Mean_Change_Pct'].abs() >= 10]
    for idx, row in large_change_features.iterrows():
        print(f"  • {row['Feature']}: {row['Mean_Change_Pct']:+.2f}% (Missing: {row['Missing_Pct']:.1f}%)")

# Overall assessment
avg_mean_change = features_with_missing['Mean_Change_Pct'].abs().mean()
avg_std_change = features_with_missing['Std_Change_Pct'].abs().mean()

print(f"\nOverall Statistics:")
print(f"  Average absolute mean change: {avg_mean_change:.2f}%")
print(f"  Average absolute std change: {avg_std_change:.2f}%")

if avg_mean_change < 5:
    quality = "EXCELLENT"
    symbol = "✅"
elif avg_mean_change < 10:
    quality = "GOOD"
    symbol = "✓"
else:
    quality = "FAIR"
    symbol = "⚠️"

print(f"\n{symbol} Imputation Quality: {quality}")
print(f"   (Average mean change: {avg_mean_change:.2f}%)")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print(" IMPUTATION VALIDATION COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 Data Overview:")
print(f"   - Total features: {len(X_original.columns)}")
print(f"   - Features with missing values: {len(features_with_missing)}")
print(f"   - Total missing values imputed: {X_original.isnull().sum().sum()}")
print(f"   - Overall missing percentage: {X_original.isnull().sum().sum() / X_original.size * 100:.2f}%")

print(f"\n🔍 Distribution Preservation:")
print(f"   - Features with preserved distribution: {preserved_count}/{len(ks_df)}")
print(f"   - Preservation rate: {preserved_count/len(ks_df)*100:.1f}%")

print(f"\n📈 Statistical Changes:")
print(f"   - Small changes (<5%): {small_change} features")
print(f"   - Moderate changes (5-10%): {moderate_change} features")
print(f"   - Large changes (>10%): {large_change} features")
print(f"   - Average mean change: {avg_mean_change:.2f}%")
print(f"   - Average std change: {avg_std_change:.2f}%")

print(f"\n🎯 Quality Rating: {quality}")

print(f"\n📁 Output Files:")
print(f"   - Detailed comparison: {comparison_path}")
print(f"   - Missing features: {missing_comparison_path}")
print(f"   - KS test results: {ks_path}")
print(f"   - Visualizations: {plots_dir}/ (5 plots)")

print(f"\n💡 Conclusion:")
if avg_mean_change < 5 and preserved_count / len(ks_df) > 0.8:
    print(f"   ✅ KNN imputation successfully preserved data distributions!")
    print(f"   The imputed values closely match the original data characteristics.")
elif avg_mean_change < 10:
    print(f"   ✓ KNN imputation performed well overall.")
    print(f"   Minor changes detected, but generally acceptable.")
else:
    print(f"   ⚠️ Some features show notable changes after imputation.")
    print(f"   Consider reviewing features with large changes.")

print("\n" + "="*80)
print("✓ Imputation Validation Complete!")
print("="*80)
