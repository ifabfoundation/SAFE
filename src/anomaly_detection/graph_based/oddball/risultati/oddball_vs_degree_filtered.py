#!/usr/bin/env python
# coding: utf-8

# Plot Oddball Score vs Degree Distribution (Filtered - Remove Top 2 Anomalous Nodes) 
# for NVG_p3_temp

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pickle

# Paths
oddball_file = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores/G_NVG_p3_temp_oddball_results.csv')
degree_file = Path('/home/projects/safe/outputs/networks/degree/degree_distributions.pkl')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores/post_analisi')
output_dir.mkdir(parents=True, exist_ok=True)

# Load oddball results
print(f"Loading oddball results from: {oddball_file}")
if not oddball_file.exists():
    print(f"Error: Oddball file not found: {oddball_file}")
    exit(1)

df_oddball = pd.read_csv(oddball_file)
print(f"Loaded {len(df_oddball)} nodes with oddball scores")

# Load degree distributions
print(f"\nLoading degree distributions from: {degree_file}")
if not degree_file.exists():
    print(f"Error: Degree file not found: {degree_file}")
    exit(1)

with open(degree_file, 'rb') as f:
    degree_data = pickle.load(f)

# Find G_NVG_p3_temp in the degree data
if 'G_NVG_p3_temp' not in degree_data:
    print(f"Error: G_NVG_p3_temp not found in degree data")
    print(f"Available keys: {list(degree_data.keys())}")
    exit(1)

degree_info = degree_data['G_NVG_p3_temp']

# Check structure
if isinstance(degree_info, dict) and 'degrees' in degree_info:
    # New structure: {'degrees': [...], 'degree_counts': [...]}
    degrees_list = degree_info['degrees']
    # Create node-degree mapping (assuming node index = list index)
    degree_dict = {i: deg for i, deg in enumerate(degrees_list)}
else:
    # Old structure: direct {node: degree} dict
    degree_dict = degree_info

print(f"Loaded degree data for {len(degree_dict)} nodes")

# Get top 2 nodes by oddball score
top_2_oddball = df_oddball.nlargest(2, 'oddball_score')
top_2_nodes = top_2_oddball['node'].values.astype(int)

print(f"\nTop 2 nodes with highest oddball scores:")
for idx, row in top_2_oddball.iterrows():
    node = int(row['node'])
    # Get degree from degree_dict
    node_degree = degree_dict.get(node, 'N/A')
    print(f"  Node {node}: oddball_score = {row['oddball_score']:.6f}, degree = {node_degree}")

# Create filtered degree dict (remove top 2 nodes)
degree_dict_filtered = {k: v for k, v in degree_dict.items() if k not in top_2_nodes}
print(f"\nOriginal degree dict: {len(degree_dict)} nodes")
print(f"Filtered degree dict: {len(degree_dict_filtered)} nodes (removed {len(degree_dict) - len(degree_dict_filtered)})")

# Create dataframe with filtered degrees
df_degree_filtered = pd.DataFrame(list(degree_dict_filtered.items()), columns=['node', 'degree'])
df_degree_filtered['node'] = df_degree_filtered['node'].astype(int)

# Merge with oddball scores (filter oddball to match)
df_oddball_filtered = df_oddball[~df_oddball['node'].isin(top_2_nodes)].copy()
df_oddball_filtered['node'] = df_oddball_filtered['node'].astype(int)

df_merged = pd.merge(df_oddball_filtered, df_degree_filtered, on='node', how='inner')
print(f"Merged filtered data: {len(df_merged)} nodes")

# Use degree_y (from pickle) since degree_x is from oddball CSV
degree_col = 'degree_y' if 'degree_y' in df_merged.columns else 'degree'

# Create scatter plot (only filtered data)
print("\nCreating oddball vs degree scatter plot (filtered)...")
plt.figure(figsize=(12, 8))

# Plot filtered data only
plt.scatter(df_merged[degree_col], df_merged['oddball_score'], 
            alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)

plt.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.3)

plt.xlabel('Degree', fontsize=14)
plt.ylabel('Oddball Score', fontsize=14)
plt.title('Oddball Score vs Degree (Top 2 Anomalous Nodes Removed) - NVG_p3_temp', 
          fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plot_file = output_dir / 'oddball_vs_degree_filtered.png'
plt.savefig(plot_file, dpi=200, bbox_inches='tight')
plt.close()

print(f"\n✓ Plot saved to: {plot_file}")

# Save filtered data statistics
stats_file = output_dir / 'oddball_vs_degree_filtered_stats.txt'
with open(stats_file, 'w') as f:
    f.write("Oddball vs Degree Analysis (Filtered) - NVG_p3_temp\n")
    f.write("="*60 + "\n\n")
    
    f.write("Removed Nodes (Top 2 Oddball Scores):\n")
    f.write("-"*40 + "\n")
    for node in top_2_nodes:
        node_oddball = df_oddball[df_oddball['node'] == node]['oddball_score'].values[0]
        node_degree = degree_dict.get(node, 'N/A')
        f.write(f"Node {node}: oddball={node_oddball:.6f}, degree={node_degree}\n")
    
    f.write("\n\nFiltered Data Statistics:\n")
    f.write("-"*40 + "\n")
    f.write(f"Total nodes: {len(df_merged)}\n")
    f.write(f"Degree range: [{df_merged[degree_col].min()}, {df_merged[degree_col].max()}]\n")
    f.write(f"Degree mean: {df_merged[degree_col].mean():.2f}\n")
    f.write(f"Degree std: {df_merged[degree_col].std():.2f}\n")
    f.write(f"\nOddball score range: [{df_merged['oddball_score'].min():.6f}, {df_merged['oddball_score'].max():.6f}]\n")
    f.write(f"Oddball score mean: {df_merged['oddball_score'].mean():.6f}\n")
    f.write(f"Oddball score std: {df_merged['oddball_score'].std():.6f}\n")
    
    # Correlation
    correlation = df_merged[degree_col].corr(df_merged['oddball_score'])
    f.write(f"\nCorrelation (degree vs oddball): {correlation:.4f}\n")

print(f"✓ Statistics saved to: {stats_file}")
print("\nDone!")
