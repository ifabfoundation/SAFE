# Plot Closeness Centrality vs Oddball Score for each graph

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
closeness_dir = Path('/home/projects/safe/outputs/networks/centralities/closeness')
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores')
output_dir = Path('/home/projects/safe/outputs/networks/centralities/plots/closeness_vs_oddball')
output_dir.mkdir(parents=True, exist_ok=True)

print("Finding closeness CSV files...")
print(f"Closeness directory: {closeness_dir}")

# Find all closeness CSV files
closeness_files = list(closeness_dir.glob('*_closeness.csv'))
print(f"Found {len(closeness_files)} closeness files")

# Extract graph names from closeness files
graph_names = [f.stem.replace('_closeness', '') for f in closeness_files]
print(f"Graphs with closeness: {graph_names}")

print("\n" + "=" * 60)
print("PLOTTING CLOSENESS VS ODDBALL SCORE")
print("=" * 60)

# Process each graph
for graph_name in graph_names:
    print(f"\nProcessing {graph_name}...")
    
    # Load closeness data from CSV
    closeness_file = closeness_dir / f"{graph_name}_closeness.csv"
    
    if not closeness_file.exists():
        print(f"  Warning: Closeness file not found: {closeness_file.name}")
        continue
    
    df_closeness = pd.read_csv(closeness_file)
    print(f"  Loaded closeness: {len(df_closeness)} nodes")
    
    # Build oddball filename
    oddball_file = oddball_dir / f"{graph_name}_oddball_results.csv"
    
    if not oddball_file.exists():
        print(f"  Warning: Oddball file not found: {oddball_file.name}")
        continue
    
    # Load oddball data
    df_oddball = pd.read_csv(oddball_file)
    print(f"  Loaded oddball: {len(df_oddball)} nodes")
    
    # Merge on node index
    df_merged = pd.merge(df_closeness, df_oddball[['node', 'oddball_score']], on='node', how='inner')
    
    if len(df_merged) == 0:
        print(f"  Warning: No matching nodes between closeness and oddball data")
        continue
    
    print(f"  Merged {len(df_merged)} nodes")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df_merged['closeness_centrality'], 
                        df_merged['oddball_score'],
                        alpha=0.6, 
                        s=30,
                        c=df_merged['node'],
                        cmap='viridis',
                        edgecolors='black',
                        linewidth=0.3)
    
    ax.set_xlabel('Closeness Centrality', fontsize=12)
    ax.set_ylabel('Oddball Score', fontsize=12)
    ax.set_title(f'Closeness vs Oddball Score - {graph_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar with clear label
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Node Index\n(position in time series)', fontsize=11, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    
    # Save plot
    plot_file = output_dir / f"{graph_name}_closeness_vs_oddball.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {plot_file.name}")

print("\n" + "=" * 60)
print("PLOTTING COMPLETED")
print("=" * 60)
print(f"Output directory: {output_dir}")
print("Done!")
