#!/usr/bin/env python
# coding: utf-8

# Plot Top 40 Oddball Nodes (Lambda - λ_i ~ W_i^beta) on Original Time Series - Bonfiglioli

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores/lambda_scores')
data_file = Path('/home/projects/safe/data/bonfiglioli/processed/ETR_946 - Prova 30_5 - 310 - Fatica 336h_1_1.csv')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_vs_timeseries/dominantpair')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading Bonfiglioli dataset...")
df_data = pd.read_csv(data_file, index_col=0)
print(f"Dataset shape: {df_data.shape}")
print(f"Columns: {df_data.columns.tolist()}")

print("\nFinding oddball CSV files (NVG only)...")
oddball_files = sorted([f for f in oddball_dir.glob('*_oddball_lambda_results.csv') if 'G_NVG_' in f.stem])
print(f"Found {len(oddball_files)} oddball files (NVG)")

# Target columns for Bonfiglioli
target_keys = [ 'vib_rid_marcia_rms']

# Map graph names to column names
# Graph format: G_NVG_temp_cassa_riduttore
# Column format: temp_cassa_riduttore
def graph_to_column(graph_name):
    """
    Extract column name from graph name.
    G_NVG_temp_cassa_riduttore -> temp_cassa_riduttore
    """
    # Remove G_NVG_ or G_HVG_ prefix
    if graph_name.startswith('G_NVG_'):
        col = graph_name[6:]  # Remove 'G_NVG_'
    elif graph_name.startswith('G_HVG_'):
        col = graph_name[6:]  # Remove 'G_HVG_'
    else:
        col = graph_name
    
    return col

print("\n" + "=" * 60)
print("PLOTTING TOP 40 ODDBALL NODES ON TIME SERIES")
print("=" * 60)

processed = 0
skipped = 0

# Process each graph
for oddball_file in oddball_files:
    graph_name = oddball_file.stem.replace('_oddball_lambda_results', '')
    print(f"\nProcessing {graph_name}...")
    
    # Get corresponding column name
    col_name = graph_to_column(graph_name)
    print(f"  Column name: {col_name}")
    
    if col_name not in df_data.columns:
        print(f"  Warning: Column '{col_name}' not found in dataset")
        print(f"  Available columns (first 5): {df_data.columns.tolist()[:5]}")
        skipped += 1
        continue
    
    try:
        # Load oddball results
        df_oddball = pd.read_csv(oddball_file)
        
        # Get top 40 nodes by oddball score
        top_40 = df_oddball.nlargest(40, 'oddball_score')
        print(f"  Top 40 oddball scores: [{top_40['oddball_score'].values[0]:.3f}, ..., {top_40['oddball_score'].values[-1]:.3f}]")
        print(f"  Top 40 node indices (first 10): {top_40['node'].values[:10].astype(int).tolist()}")
        
        # Get time series data
        ts_data = df_data[col_name].values
        time_indices = np.arange(len(ts_data))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot time series
        ax.plot(time_indices, ts_data, color='gray', alpha=0.7, linewidth=0.8, label='Time series')
        
        # Highlight top 40 oddball nodes with colormap
        top_nodes = top_40['node'].values.astype(int)
        top_scores = top_40['oddball_score'].values
        
        scatter = ax.scatter(top_nodes, ts_data[top_nodes], 
                            c=top_scores, cmap='hot_r', s=80, alpha=0.9, 
                            edgecolors='black', linewidth=1, zorder=5,
                            vmin=top_scores.min(), vmax=top_scores.max())
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Oddball Score', fontsize=11)
        
        ax.set_xlabel('Node Index (Time)', fontsize=12)
        ax.set_ylabel(f'{col_name}', fontsize=12)
        ax.set_title(f'Top 40 Oddball Nodes - {graph_name}\n(weights: distance, oddball: λ_i ~ W_i^β)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / f"{graph_name}_lambda_vs_timeseries.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {plot_file.name}")
        processed += 1
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        skipped += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Total files: {len(oddball_files)}")
print(f"Processed: {processed}")
print(f"Skipped: {skipped}")
print(f"\nPlots saved to: {output_dir}")
print("\nDone!")
