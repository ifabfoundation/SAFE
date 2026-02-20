#!/usr/bin/env python
# coding: utf-8

# Plot Top 2 Oddball Nodes for NVG_p3_temp on Time Series (Oddball Score Only)

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

# Get top 2 nodes by oddball score
top_2_oddball = df_oddball.nlargest(2, 'oddball_score')
print(f"\nTop 2 nodes with highest oddball scores:")
for idx, row in top_2_oddball.iterrows():
    print(f"  Node {int(row['node'])}: oddball_score = {row['oddball_score']:.6f}")

# Load time series data
print(f"\nLoading time series from: {data_file}")
df_data = pd.read_csv(data_file, index_col=0)

# Get p3_temp column
temp_cols = [col for col in df_data.columns if 'p3' in col and 'temp' in col]
print(f"Available p3 temp columns: {temp_cols}")

if len(temp_cols) == 0:
    print("Error: No p3_temp column found")
    exit(1)

col_name = temp_cols[0]
print(f"Using column: {col_name}")

ts_data = df_data[col_name].values
time_indices = np.arange(len(ts_data))

# Create single plot: Time series with top 2 oddball nodes highlighted
print("\nCreating time series plot with top 2 oddball nodes...")
plt.figure(figsize=(16, 6))

plt.plot(time_indices, ts_data, color='blue', alpha=0.6, linewidth=0.8, label='Time series')

# Highlight top 2 nodes
colors = ['red', 'orange']
for i, (idx, row) in enumerate(top_2_oddball.iterrows()):
    node = int(row['node'])
    oddball_score = row['oddball_score']
    
    plt.scatter(node, ts_data[node], 
                color=colors[i], s=150, alpha=0.9, 
                edgecolors='black', linewidth=2,
                label=f'Node {node} (score={oddball_score:.4f})', zorder=5)
    
    plt.annotate(f"Node {node}\n(oddball: {oddball_score:.4f})",
                xy=(node, ts_data[node]),
                xytext=(20, 20), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, 
                         edgecolor=colors[i], linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                               lw=2, color=colors[i]))

plt.xlabel('Node Index (Time)', fontsize=14)
plt.ylabel(f'{col_name}', fontsize=14)
plt.title(f'Top 2 Nodes by Oddball Score - NVG_p3_temp', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plot_file = output_dir / 'oddball_p3_temp_top2_simple.png'
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✓ Plot saved to: {plot_file}")

# Save top 2 nodes info to text file
info_file = output_dir / 'oddball_p3_temp_top2_nodes_simple.txt'
with open(info_file, 'w') as f:
    f.write("Top 2 Nodes by Oddball Score for NVG_p3_temp\n")
    f.write("="*50 + "\n\n")
    for idx, row in top_2_oddball.iterrows():
        node = int(row['node'])
        oddball = row['oddball_score']
        value = ts_data[node]
        f.write(f"Node {node}:\n")
        f.write(f"  Oddball Score: {oddball:.6f}\n")
        f.write(f"  Time Series Value: {value:.4f}\n")
        f.write(f"  Egonet Nodes: {int(row['egonet_nodes'])}\n")
        f.write(f"  Egonet Edges: {int(row['egonet_edges'])}\n")
        f.write(f"  Egonet Total Weight: {row['egonet_total_weight']:.6f}\n")
        if 'degree' in row:
            f.write(f"  Degree: {int(row['degree'])}\n")
        f.write("\n")

print(f"✓ Node info saved to: {info_file}")
print("\nDone!")
