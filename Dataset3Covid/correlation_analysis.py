"""
COVID-19 Feature Correlation Analysis
Visualizes correlation between features and SARS-CoV-2 target
Identifies most correlated features for feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pointbiserialr, spearmanr, pearsonr

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print(" COVID-19 FEATURE CORRELATION ANALYSIS")
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
df = pd.read_csv(dataset_path)
print(f"✓ Dataset loaded: {df.shape}")

# ============================================================================
# 2. CALCULATE CORRELATIONS
# ============================================================================

print("\n2. CALCULATING CORRELATIONS WITH TARGET")
print("="*80)

# Separate features and target
X = df.drop('SARSCov', axis=1)
y = df['SARSCov']

print(f"Features: {X.shape[1]}")
print(f"Target: SARSCov (Binary: 0=Negative, 1=Positive)")

# Calculate different types of correlations
correlation_results = []

for col in X.columns:
    # Remove missing values for this feature
    valid_mask = X[col].notna() & y.notna()
    
    if valid_mask.sum() > 0:
        x_valid = X.loc[valid_mask, col]
        y_valid = y[valid_mask]
        
        # Pearson correlation (linear)
        pearson_corr, pearson_pval = pearsonr(x_valid, y_valid)
        
        # Spearman correlation (monotonic, rank-based)
        spearman_corr, spearman_pval = spearmanr(x_valid, y_valid)
        
        # Point-biserial correlation (continuous vs binary)
        pointbiserial_corr, pointbiserial_pval = pointbiserialr(y_valid, x_valid)
        
        # Calculate additional statistics
        mean_negative = x_valid[y_valid == 0].mean()
        mean_positive = x_valid[y_valid == 1].mean()
        diff_means = mean_positive - mean_negative
        
        correlation_results.append({
            'Feature': col,
            'Pearson_r': pearson_corr,
            'Pearson_pval': pearson_pval,
            'Spearman_r': spearman_corr,
            'Spearman_pval': spearman_pval,
            'PointBiserial_r': pointbiserial_corr,
            'PointBiserial_pval': pointbiserial_pval,
            'Mean_Negative': mean_negative,
            'Mean_Positive': mean_positive,
            'Diff_Means': diff_means,
            'Valid_Samples': valid_mask.sum()
        })

# Create DataFrame
corr_df = pd.DataFrame(correlation_results)

# Sort by absolute Pearson correlation
corr_df['Abs_Pearson'] = corr_df['Pearson_r'].abs()
corr_df = corr_df.sort_values('Abs_Pearson', ascending=False)

print("\n✓ Correlations calculated for all features")
print(f"Total features analyzed: {len(corr_df)}")

# ============================================================================
# 3. TOP CORRELATED FEATURES
# ============================================================================

print("\n3. TOP CORRELATED FEATURES")
print("="*80)

# Filter significant correlations (p-value < 0.05)
significant_df = corr_df[corr_df['Pearson_pval'] < 0.05].copy()

print(f"\nSignificant correlations (p < 0.05): {len(significant_df)}")
print("\nTop 20 Features by Absolute Correlation:")
print("="*80)

display_cols = ['Feature', 'Pearson_r', 'Spearman_r', 'PointBiserial_r', 'Pearson_pval']
print(corr_df[display_cols].head(20).to_string(index=False))

# ============================================================================
# 4. FEATURE ANALYSIS BY CORRELATION STRENGTH
# ============================================================================

print("\n4. FEATURE ANALYSIS BY CORRELATION STRENGTH")
print("="*80)

# Positive correlations (increase with COVID-19)
positive_corr = corr_df[corr_df['Pearson_r'] > 0].sort_values('Pearson_r', ascending=False)
print(f"\nPositive correlations: {len(positive_corr)}")
if len(positive_corr) > 0:
    print("\nTop 10 Positively Correlated (higher in COVID-19 positive):")
    print(positive_corr[['Feature', 'Pearson_r', 'Mean_Negative', 'Mean_Positive']].head(10).to_string(index=False))

# Negative correlations (decrease with COVID-19)
negative_corr = corr_df[corr_df['Pearson_r'] < 0].sort_values('Pearson_r', ascending=True)
print(f"\nNegative correlations: {len(negative_corr)}")
if len(negative_corr) > 0:
    print("\nTop 10 Negatively Correlated (lower in COVID-19 positive):")
    print(negative_corr[['Feature', 'Pearson_r', 'Mean_Negative', 'Mean_Positive']].head(10).to_string(index=False))

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================

print("\n5. GENERATING VISUALIZATIONS")
print("="*80)

# 5.1 Correlation Bar Plot (Top 20)
plt.figure(figsize=(12, 10))
top_20 = corr_df.head(20)
colors = ['red' if x < 0 else 'green' for x in top_20['Pearson_r']]
bars = plt.barh(range(len(top_20)), top_20['Pearson_r'], color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
plt.yticks(range(len(top_20)), top_20['Feature'])
plt.xlabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Top 20 Features Correlated with COVID-19 Status', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=2)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_20['Pearson_r'])):
    x_pos = value + (0.01 if value > 0 else -0.01)
    ha = 'left' if value > 0 else 'right'
    plt.text(x_pos, i, f'{value:.3f}', va='center', ha=ha, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_top20.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_top20.png")

# 5.2 All Features Correlation Heatmap-style
plt.figure(figsize=(14, max(10, len(corr_df)*0.3)))
sorted_features = corr_df['Feature'].tolist()
sorted_corr = corr_df['Pearson_r'].tolist()

# Create color map
colors = plt.cm.RdYlGn([(x + 1) / 2 for x in sorted_corr])

bars = plt.barh(range(len(sorted_features)), sorted_corr, color=colors, edgecolor='black', linewidth=0.5)
plt.yticks(range(len(sorted_features)), sorted_features, fontsize=8)
plt.xlabel('Pearson Correlation with COVID-19 Status', fontsize=12, fontweight='bold')
plt.title('All Features Correlation with COVID-19', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=2)
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=-1, vmax=1))
sm.set_array([])
cbar = plt.colorbar(sm, ax=plt.gca(), orientation='vertical', pad=0.01)
cbar.set_label('Correlation Coefficient', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_all_features.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_all_features.png")

# 5.3 Correlation Comparison (Pearson vs Spearman)
plt.figure(figsize=(10, 8))
plt.scatter(corr_df['Pearson_r'], corr_df['Spearman_r'], 
           c=corr_df['Abs_Pearson'], cmap='viridis', s=100, alpha=0.6, edgecolors='black')
plt.xlabel('Pearson Correlation', fontsize=12, fontweight='bold')
plt.ylabel('Spearman Correlation', fontsize=12, fontweight='bold')
plt.title('Pearson vs Spearman Correlation Comparison', fontsize=14, fontweight='bold')
plt.plot([-1, 1], [-1, 1], 'r--', linewidth=2, label='Perfect Agreement')
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.grid(alpha=0.3)
plt.legend()
cbar = plt.colorbar(label='Absolute Correlation Strength')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_pearson_vs_spearman.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_pearson_vs_spearman.png")

# 5.4 Correlation Heatmap (Top 15 vs Top 15)
top_15_features = corr_df.head(15)['Feature'].tolist()

# Create correlation matrix for top features
df_top = df[top_15_features + ['SARSCov']]
correlation_matrix = df_top.corr()

plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": .8}, 
            mask=mask, vmin=-1, vmax=1)
plt.title('Correlation Matrix: Top 15 Features + Target', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_matrix_top15.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_matrix_top15.png")

# 5.5 P-value significance plot
plt.figure(figsize=(12, 8))
log_pvals = -np.log10(corr_df['Pearson_pval'] + 1e-300)  # Add small value to avoid log(0)
colors_sig = ['green' if p < 0.05 else 'red' for p in corr_df['Pearson_pval']]

plt.scatter(corr_df['Pearson_r'], log_pvals, c=colors_sig, s=100, alpha=0.6, edgecolors='black')
plt.axhline(y=-np.log10(0.05), color='blue', linestyle='--', linewidth=2, label='p = 0.05')
plt.axhline(y=-np.log10(0.01), color='orange', linestyle='--', linewidth=2, label='p = 0.01')
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
plt.ylabel('-log10(p-value)', fontsize=12, fontweight='bold')
plt.title('Correlation Significance: Volcano Plot', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# Annotate top features
for idx, row in corr_df.head(10).iterrows():
    plt.annotate(row['Feature'], 
                xy=(row['Pearson_r'], -np.log10(row['Pearson_pval'] + 1e-300)),
                xytext=(5, 5), textcoords='offset points', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_significance_volcano.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_significance_volcano.png")

# 5.6 Mean differences plot (COVID+ vs COVID-)
top_diff = corr_df.nlargest(15, 'Abs_Pearson')

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(top_diff))
width = 0.35

bars1 = ax.bar(x - width/2, top_diff['Mean_Negative'], width, label='COVID-19 Negative', 
               color='green', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x + width/2, top_diff['Mean_Positive'], width, label='COVID-19 Positive', 
               color='red', alpha=0.7, edgecolor='black')

ax.set_xlabel('Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Mean Value', fontsize=12, fontweight='bold')
ax.set_title('Feature Mean Values: COVID-19 Negative vs Positive', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(top_diff['Feature'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_mean_differences.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_mean_differences.png")

# 5.7 Absolute correlation ranking
plt.figure(figsize=(12, 10))
top_30_abs = corr_df.head(30)
colors_grad = plt.cm.plasma(np.linspace(0, 1, len(top_30_abs)))

bars = plt.barh(range(len(top_30_abs)), top_30_abs['Abs_Pearson'], 
               color=colors_grad, edgecolor='black', linewidth=1)
plt.yticks(range(len(top_30_abs)), top_30_abs['Feature'])
plt.xlabel('Absolute Correlation (|r|)', fontsize=12, fontweight='bold')
plt.title('Top 30 Features by Absolute Correlation Strength', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, value) in enumerate(zip(bars, top_30_abs['Abs_Pearson'])):
    plt.text(value + 0.005, i, f'{value:.3f}', va='center', fontweight='bold', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(plots_dir, 'correlation_absolute_ranking.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: correlation_absolute_ranking.png")

print(f"\n✓ All visualizations saved to: {plots_dir}/")

# ============================================================================
# 6. SAVE CORRELATION RESULTS
# ============================================================================

print("\n6. SAVING RESULTS")
print("="*80)

# Save full correlation table
results_path = os.path.join(output_dir, 'feature_correlations.csv')
corr_df.to_csv(results_path, index=False)
print(f"✓ Full results saved: {results_path}")

# Save significant correlations only
significant_path = os.path.join(output_dir, 'significant_correlations.csv')
significant_df.to_csv(significant_path, index=False)
print(f"✓ Significant correlations saved: {significant_path}")

# ============================================================================
# 7. FEATURE SELECTION RECOMMENDATIONS
# ============================================================================

print("\n7. FEATURE SELECTION RECOMMENDATIONS")
print("="*80)

# Strong correlations (|r| > 0.3)
strong_corr = corr_df[corr_df['Abs_Pearson'] > 0.3]
print(f"\nStrong correlations (|r| > 0.3): {len(strong_corr)} features")
if len(strong_corr) > 0:
    print("\nStrongly correlated features:")
    for idx, row in strong_corr.iterrows():
        direction = "↑ Higher" if row['Pearson_r'] > 0 else "↓ Lower"
        print(f"  • {row['Feature']}: r={row['Pearson_r']:.3f} ({direction} in COVID+)")

# Moderate correlations (0.2 < |r| < 0.3)
moderate_corr = corr_df[(corr_df['Abs_Pearson'] > 0.2) & (corr_df['Abs_Pearson'] <= 0.3)]
print(f"\nModerate correlations (0.2 < |r| ≤ 0.3): {len(moderate_corr)} features")
if len(moderate_corr) > 0:
    print("\nModerately correlated features:")
    for idx, row in moderate_corr.iterrows():
        direction = "↑ Higher" if row['Pearson_r'] > 0 else "↓ Lower"
        print(f"  • {row['Feature']}: r={row['Pearson_r']:.3f} ({direction} in COVID+)")

# Weak correlations (|r| < 0.2)
weak_corr = corr_df[corr_df['Abs_Pearson'] <= 0.2]
print(f"\nWeak correlations (|r| ≤ 0.2): {len(weak_corr)} features")
print("(These features may have limited predictive power individually)")

# ============================================================================
# 8. SUMMARY
# ============================================================================

print("\n" + "="*80)
print(" CORRELATION ANALYSIS COMPLETE - SUMMARY")
print("="*80)

print(f"\n📊 Analysis Summary:")
print(f"   - Total features analyzed: {len(corr_df)}")
print(f"   - Significant correlations (p < 0.05): {len(significant_df)}")
print(f"   - Strong correlations (|r| > 0.3): {len(strong_corr)}")
print(f"   - Moderate correlations (0.2 < |r| ≤ 0.3): {len(moderate_corr)}")
print(f"   - Weak correlations (|r| ≤ 0.2): {len(weak_corr)}")

if len(corr_df) > 0:
    print(f"\n🔍 Top 3 Most Correlated Features:")
    for i, (idx, row) in enumerate(corr_df.head(3).iterrows(), 1):
        direction = "positively" if row['Pearson_r'] > 0 else "negatively"
        print(f"   {i}. {row['Feature']}: r={row['Pearson_r']:.4f} ({direction} correlated)")

print(f"\n📁 Output Files:")
print(f"   - Full correlations: {results_path}")
print(f"   - Significant only: {significant_path}")
print(f"   - Visualizations: {plots_dir}/ (7 plots)")

print(f"\n💡 Recommendation:")
print(f"   Consider using the top {min(10, len(strong_corr) + len(moderate_corr))} correlated features")
print(f"   for feature selection to improve model efficiency.")

print("\n" + "="*80)
print("✓ Correlation Analysis Complete!")
print("="*80)
