#!/usr/bin/env python
# coding: utf-8

# Plot Top 2 Nodes with Combined Oddball+LOF Score for NVG_p3_temp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

# Paths
oddball_file = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores/G_NVG_p3_temp_oddball_results.csv')
data_file = Path('/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores/post_analisi')
output_dir.mkdir(parents=True, exist_ok=True)

# Load oddball results
print(f"Loading oddball results from: {oddball_file}")
if not oddball_file.exists():
    print(f"Error: Oddball file not found: {oddball_file}")
    exit(1)

df_oddball = pd.read_csv(oddball_file)
print(f"Loaded {len(df_oddball)} nodes with oddball scores")

# Calculate LOF scores
print("\nCalculating LOF scores...")
# Use egonet features for LOF
X = df_oddball[['egonet_nodes', 'egonet_edges']].values

lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
lof.fit_predict(X)
lof_scores_normalized = -lof.negative_outlier_factor_  # Higher = more outlier

# Add LOF scores to dataframe
df_oddball['lof_score'] = lof_scores_normalized

# Normalize both scores to [0, 1] for combination
oddball_norm = (df_oddball['oddball_score'] - df_oddball['oddball_score'].min()) / (df_oddball['oddball_score'].max() - df_oddball['oddball_score'].min())
lof_norm = (df_oddball['lof_score'] - df_oddball['lof_score'].min()) / (df_oddball['lof_score'].max() - df_oddball['lof_score'].min())

# Combined score: average of normalized oddball and LOF
df_oddball['combined_score'] = (oddball_norm + lof_norm) / 2

print(f"LOF scores calculated. Range: [{df_oddball['lof_score'].min():.4f}, {df_oddball['lof_score'].max():.4f}]")
print(f"Combined scores calculated. Range: [{df_oddball['combined_score'].min():.4f}, {df_oddball['combined_score'].max():.4f}]")

# Get top 2 nodes by combined score
top_2 = df_oddball.nlargest(2, 'combined_score')
print(f"\nTop 2 nodes with highest combined (oddball + LOF) scores:")
for idx, row in top_2.iterrows():
    print(f"  Node {int(row['node'])}:")
    print(f"    Oddball score: {row['oddball_score']:.6f}")
    print(f"    LOF score: {row['lof_score']:.6f}")
    print(f"    Combined score: {row['combined_score']:.6f}")

# Load time series data
print(f"\nLoading time series from: {data_file}")
df_data = pd.read_csv(data_file, index_col=0)

# Get p3_temp column - need to find the exact column name
# Looking for bonfi/gb1_p3_temp
temp_cols = [col for col in df_data.columns if 'p3' in col and 'temp' in col]
print(f"Available p3 temp columns: {temp_cols}")

if len(temp_cols) == 0:
    print("Error: No p3_temp column found")
    exit(1)

col_name = temp_cols[0]  # Take the first matching column
print(f"Using column: {col_name}")

ts_data = df_data[col_name].values
time_indices = np.arange(len(ts_data))

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(16, 12))

# Plot 1: Time series with top 2 combined score nodes
ax = axes[0]
ax.plot(time_indices, ts_data, color='blue', alpha=0.6, linewidth=0.8, label='Time series')

top_nodes = top_2['node'].values.astype(int)
ax.scatter(top_nodes, ts_data[top_nodes], 
           color='red', s=150, alpha=0.9, 
           edgecolors='black', linewidth=2,
           label=f'Top 2 nodes (Combined Score)', zorder=5)

for idx, row in top_2.iterrows():
    node = int(row['node'])
    combined = row['combined_score']
    ax.annotate(f"Node {node}\n(combined: {combined:.4f})",
               xy=(node, ts_data[node]),
               xytext=(20, 20), textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2),
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', lw=2, color='red'))

ax.set_xlabel('Node Index (Time)', fontsize=14)
ax.set_ylabel(f'{col_name}', fontsize=14)
ax.set_title(f'Top 2 Nodes (Oddball + LOF Combined Score) - NVG_p3_temp', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: Scatter plot of Oddball vs LOF scores
ax = axes[1]
scatter = ax.scatter(df_oddball['oddball_score'], 
                     df_oddball['lof_score'],
                     c=df_oddball['combined_score'],
                     cmap='viridis',
                     alpha=0.6,
                     s=30,
                     edgecolors='none')

# Highlight top 2 nodes
ax.scatter(top_2['oddball_score'], 
           top_2['lof_score'],
           color='red', s=200, alpha=0.9,
           edgecolors='black', linewidth=2,
           label='Top 2 nodes', zorder=5)

for idx, row in top_2.iterrows():
    node = int(row['node'])
    ax.annotate(f"Node {node}",
               xy=(row['oddball_score'], row['lof_score']),
               xytext=(10, 10), textcoords='offset points',
               fontsize=9, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
               arrowprops=dict(arrowstyle='->', lw=1.5))

ax.set_xlabel('Oddball Score', fontsize=14)
ax.set_ylabel('LOF Score', fontsize=14)
ax.set_title('Oddball Score vs LOF Score', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Combined Score', fontsize=12)

# Save plot
plot_file = output_dir / 'oddball_p3_temp_top2_combined.png'
plt.tight_layout()
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✓ Plot saved to: {plot_file}")

# Save top 2 nodes info to text file
info_file = output_dir / 'oddball_p3_temp_top2_nodes_combined.txt'
with open(info_file, 'w') as f:
    f.write("Top 2 Nodes (Oddball + LOF Combined Score) for NVG_p3_temp\n")
    f.write("="*50 + "\n\n")
    for idx, row in top_2.iterrows():
        node = int(row['node'])
        oddball = row['oddball_score']
        lof = row['lof_score']
        combined = row['combined_score']
        value = ts_data[node]
        f.write(f"Node {node}:\n")
        f.write(f"  Oddball Score: {oddball:.6f}\n")
        f.write(f"  LOF Score: {lof:.6f}\n")
        f.write(f"  Combined Score: {combined:.6f}\n")
        f.write(f"  Time Series Value: {value:.4f}\n")
        f.write(f"  Egonet Nodes: {int(row['egonet_nodes'])}\n")
        f.write(f"  Egonet Edges: {int(row['egonet_edges'])}\n")
        f.write(f"  Degree: {int(row['degree'])}\n")
        f.write("\n")

print(f"✓ Node info saved to: {info_file}")
print("\nDone!")
