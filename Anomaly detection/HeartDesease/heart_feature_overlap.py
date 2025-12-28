import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data (same as main script)
heart_df = pd.read_csv('Dataset2.csv')

# Create imbalanced dataset
normal_samples = heart_df[heart_df['target'] == 0]
anomaly_samples = heart_df[heart_df['target'] == 1]
np.random.seed(42)
anomaly_reduced = anomaly_samples.sample(n=80, random_state=42)
imbalanced_df = pd.concat([normal_samples, anomaly_reduced], ignore_index=True)

# Separate by class
normal_data = imbalanced_df[imbalanced_df['target'] == 0]
anomaly_data = imbalanced_df[imbalanced_df['target'] == 1]

# Plot feature distributions
features = ['age', 'resting bp s', 'cholesterol', 'max heart rate', 'oldpeak', 'ST slope']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(features):
    ax = axes[idx]
    
    # Plot distributions
    ax.hist(normal_data[feature], bins=20, alpha=0.6, color='blue', label=f'Normal (n={len(normal_data)})', density=True)
    ax.hist(anomaly_data[feature], bins=20, alpha=0.6, color='red', label=f'Anomaly (n={len(anomaly_data)})', density=True)
    
    # Calculate overlap
    normal_mean = normal_data[feature].mean()
    normal_std = normal_data[feature].std()
    anomaly_mean = anomaly_data[feature].mean()
    anomaly_std = anomaly_data[feature].std()
    
    ax.axvline(normal_mean, color='blue', linestyle='--', linewidth=2, label=f'Normal μ={normal_mean:.1f}')
    ax.axvline(anomaly_mean, color='red', linestyle='--', linewidth=2, label=f'Anomaly μ={anomaly_mean:.1f}')
    
    ax.set_title(f'{feature}\nMean diff: {abs(normal_mean-anomaly_mean):.1f}', fontweight='bold')
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('WHY ELLIPTIC ENVELOPE FAILS: Feature Distributions OVERLAP!\n'
             'Heart disease patients look similar to healthy patients', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('heart_feature_overlap.png', dpi=300, bbox_inches='tight')
print("Saved: heart_feature_overlap.png")

# Print statistics
print("\n" + "="*70)
print("FEATURE OVERLAP ANALYSIS")
print("="*70)
print(f"\n{'Feature':<20} {'Normal Mean':>12} {'Anomaly Mean':>12} {'Difference':>12} {'% Diff':>10}")
print("-"*70)

for feature in features:
    n_mean = normal_data[feature].mean()
    a_mean = anomaly_data[feature].mean()
    diff = abs(n_mean - a_mean)
    pct_diff = (diff / n_mean) * 100 if n_mean != 0 else 0
    print(f"{feature:<20} {n_mean:>12.2f} {a_mean:>12.2f} {diff:>12.2f} {pct_diff:>9.1f}%")

print("\n" + "="*70)
print("CONCLUSION: Heart disease features overlap significantly!")
print("Anomaly detection works best when anomalies are STATISTICAL OUTLIERS,")
print("but heart disease patients have similar vital signs to healthy people.")
print("="*70)
