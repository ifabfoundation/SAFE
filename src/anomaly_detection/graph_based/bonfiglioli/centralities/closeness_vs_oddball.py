#!/usr/bin/env python
# coding: utf-8

# Plot Oddball Scores vs Closeness Centrality for Bonfiglioli Graphs

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores')
closeness_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs/closeness/scores')
output_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs/closeness/plots')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Oddball scores directory: {oddball_dir}")
print(f"Closeness scores directory: {closeness_dir}")
print(f"Output directory: {output_dir}")

# Get all closeness score files
closeness_files = sorted(list(closeness_dir.glob('NVG_*_closeness.csv')))
print(f"\nFound {len(closeness_files)} NVG closeness score files")

if len(closeness_files) == 0:
    print("No closeness score files found for NVG graphs. Exiting.")
    exit(0)

print("\n" + "="*60)
print("PLOTTING ODDBALL SCORES VS CLOSENESS CENTRALITY")
print("="*60)

processed = 0
skipped = 0

for closeness_file in closeness_files:
    # Extract graph name from closeness filename
    # Format: NVG_colname_closeness.csv -> NVG_colname
    graph_name = closeness_file.stem.replace('_closeness', '')
    
    print(f"\nProcessing: {graph_name}")
    
    # Build oddball filename
    oddball_file = oddball_dir / f"{graph_name}_oddball_results.csv"
    
    if not oddball_file.exists():
        print(f"  Warning: Oddball file not found: {oddball_file.name}")
        skipped += 1
        continue
    
    try:
        # Load closeness data
        df_closeness = pd.read_csv(closeness_file)
        print(f"  Loaded closeness: {len(df_closeness)} nodes")
        
        # Load oddball data
        df_oddball = pd.read_csv(oddball_file)
        print(f"  Loaded oddball: {len(df_oddball)} nodes")
        
        # Merge on node index
        df_merged = pd.merge(df_closeness[['node', 'closeness']], 
                            df_oddball[['node', 'oddball_score']], 
                            on='node', how='inner')
        
        if len(df_merged) == 0:
            print(f"  Warning: No matching nodes between closeness and oddball data")
            skipped += 1
            continue
        
        print(f"  Merged data: {len(df_merged)} nodes")
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot with color based on node index
        scatter = ax.scatter(
            df_merged['closeness'],
            df_merged['oddball_score'],
            c=df_merged['node'],
            cmap='viridis',
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        # Labels and title
        ax.set_xlabel('Closeness Centrality', fontsize=12)
        ax.set_ylabel('Oddball Score', fontsize=12)
        ax.set_title(f'{graph_name}\nOddball Score vs Closeness Centrality', 
                    fontsize=14, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Node Index\n(position in time series)', fontsize=10)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Statistics text
        stats_text = (
            f'Nodes: {len(df_merged)}\n'
            f'Closeness range: [{df_merged["closeness"].min():.4f}, {df_merged["closeness"].max():.4f}]\n'
            f'Oddball range: [{df_merged["oddball_score"].min():.2f}, {df_merged["oddball_score"].max():.2f}]\n'
            f'Mean oddball: {df_merged["oddball_score"].mean():.2f}\n'
            f'Mean closeness: {df_merged["closeness"].mean():.4f}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        output_file = output_dir / f"{graph_name}_oddball_vs_closeness.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {output_file.name}")
        processed += 1
        
    except Exception as e:
        print(f"  ✗ Error processing {graph_name}: {e}")
        import traceback
        traceback.print_exc()
        skipped += 1

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total files found: {len(closeness_files)}")
print(f"Successfully processed: {processed}")
print(f"Skipped: {skipped}")
print(f"\nPlots saved to: {output_dir}")
print("\nDone!")
