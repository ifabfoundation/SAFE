#!/usr/bin/env python
# coding: utf-8

# Plot branch nodes on original time series with different colors

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
branches_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_fits/weighted_fits_with_branches')
data_file = Path('/home/projects/safe/data/bonfiglioli/processed/ETR_946 - Prova 30_5 - 310 - Fatica 336h_1_1.csv')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_vs_timeseries/branches_colored')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PLOTTING BRANCHES ON TIME SERIES")
print("=" * 80)

# Load Bonfiglioli dataset
print("\nLoading Bonfiglioli dataset...")
df_data = pd.read_csv(data_file)
df_data = df_data.drop(columns=['datetime', 'time'], errors='ignore')
print(f"Dataset shape: {df_data.shape}")

# Target columns
target_keys = ['tan_oil', 'temp_oil_marcia', 'temp_pt100_oil', 'vib_mot_marcia_rms', 
               'vib_rid_marc2_rms', 'vib_rid_marcia_rms', 'temp_cassa_riduttore', 'temp_mot_marcia']

# Graph name to column name mapping
def graph_to_column(graph_name):
    """Extract column name from graph name G_NVG_temp_oil_marcia -> temp_oil_marcia"""
    if graph_name.startswith('G_NVG_'):
        return graph_name[6:]
    return graph_name

# Find all branch info files
branch_info_files = sorted(branches_dir.glob('*_branches_info.csv'))

# Filter for G_NVG_temp_mot_marcia only (for testing)
branch_info_files = [f for f in branch_info_files if 'G_NVG_temp_mot_marcia' in f.stem]

print(f"\nFound {len(branch_info_files)} graphs with branch analysis (filtered for G_NVG_temp_mot_marcia)")

# Branch colors (must match analyze_branches.py) - now using 3 branches
branch_colors = ['orange', 'cyan', 'magenta']

for info_file in branch_info_files:
    graph_name = info_file.stem.replace('_branches_info', '')
    print(f"\n{'='*80}")
    print(f"Processing: {graph_name}")
    print(f"{'='*80}")
    
    try:
        # Get column name
        col_name = graph_to_column(graph_name)
        
        if col_name not in df_data.columns:
            print(f"  Column '{col_name}' not found in dataset, skipping...")
            continue
        
        # Get time series data (full dataset)
        ts_data = df_data[col_name].values
        time_indices = np.arange(len(ts_data))
        
        print(f"  Time series length: {len(ts_data)}")
        
        # Load branch info
        branch_info = pd.read_csv(info_file)
        n_branches = len(branch_info)
        print(f"  Number of branches: {n_branches}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # Plot time series in gray
        ax.plot(time_indices, ts_data, color='dimgray', alpha=0.5, linewidth=0.8, 
                label='Time series', zorder=1)
        
        # Plot each branch with different color
        legend_handles = []
        
        for i, row in branch_info.iterrows():
            branch_id = int(row['cluster_id'])
            
            # Load nodes for this branch
            nodes_file = branches_dir / f"{graph_name}_branch_{i}_nodes.csv"
            
            if not nodes_file.exists():
                print(f"  Warning: {nodes_file.name} not found")
                continue
            
            nodes_df = pd.read_csv(nodes_file)
            nodes = nodes_df['node'].values.astype(int)
            
            # Filter valid indices
            valid_nodes = nodes[nodes < len(ts_data)]
            
            if len(valid_nodes) == 0:
                print(f"  Branch {i}: No valid nodes")
                continue
            
            # Plot nodes for this branch
            color = branch_colors[i % len(branch_colors)]
            scatter = ax.scatter(valid_nodes, ts_data[valid_nodes],
                               c=color, s=30, alpha=0.7,
                               edgecolors='black', linewidth=0.3,
                               label=f'Branch {i}: β={row["beta"]:.4f}, int={row["intercept"]:.4f} ({len(valid_nodes)} pts)',
                               zorder=5)
            
            print(f"  Branch {i}: {len(valid_nodes)} nodes plotted")
        
        # Labels and title
        ax.set_xlabel('Node Index (Time)', fontsize=12)
        ax.set_ylabel(col_name, fontsize=12)
        title = f'{graph_name}\nBranches on Time Series (colored by power law branch)'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / f"{graph_name}_branches_on_timeseries.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {plot_file.name}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("PLOTTING COMPLETE")
print("=" * 80)
print(f"Output directory: {output_dir}")
