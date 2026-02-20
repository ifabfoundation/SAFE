#!/usr/bin/env python
# coding: utf-8

# Plot Oddball Scores (Weighted - W_i ~ E_i^beta) vs Degree for Weighted Graphs
#with 'distance' weights

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/weighted/oddball_scores')
degrees_dir = Path('/home/projects/safe/outputs/networks/grafi/weighted/degrees')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/weighted/plots/oddball_vs_degree')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Oddball scores directory: {oddball_dir}")
print(f"Degrees directory: {degrees_dir}")
print(f"Output directory: {output_dir}")

# Get all oddball score files (NVG only)
oddball_files = sorted([f for f in oddball_dir.glob('*_oddball_weighted_results.csv') if 'G_NVG_' in f.stem])
print(f"\nFound {len(oddball_files)} oddball score files (NVG)")

if len(oddball_files) == 0:
    print("No oddball score files found. Exiting.")
    exit(0)

print("\n" + "="*60)
print("PLOTTING ODDBALL SCORES (W_i ~ E_i^β) VS DEGREE")
print("="*60)

processed = 0
skipped = 0

for oddball_file in oddball_files:
    # Extract graph name from oddball filename
    # Format: G_NVG_bonfi_gb1_p3_acc_cfa_oddball_weighted_results.csv
    # -> G_NVG_bonfi_gb1_p3_acc_cfa
    graph_name = oddball_file.stem.replace('_oddball_weighted_results', '')
    
    print(f"\nProcessing: {graph_name}")
    
    # Build degree filename
    degree_file = degrees_dir / f"{graph_name}_degrees.csv"
    
    if not degree_file.exists():
        print(f"  Warning: Degree file not found: {degree_file.name}")
        skipped += 1
        continue
    
    try:
        # Load oddball data
        df_oddball = pd.read_csv(oddball_file)
        print(f"  Loaded oddball: {len(df_oddball)} nodes")
        
        # Load degree data
        df_degree = pd.read_csv(degree_file)
        print(f"  Loaded degrees: {len(df_degree)} nodes")
        
        # Merge on node index
        df_merged = pd.merge(df_oddball[['node', 'oddball_score']], df_degree[['node', 'degree']], 
                            on='node', how='inner')
        
        if len(df_merged) == 0:
            print(f"  Warning: No matching nodes between oddball and degree data")
            skipped += 1
            continue
        
        print(f"  Merged data: {len(df_merged)} nodes")
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Simple scatter plot
        ax.scatter(
            df_merged['degree'],
            df_merged['oddball_score'],
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
        
        # Labels and title
        ax.set_xlabel('Degree', fontsize=12)
        ax.set_ylabel('Oddball Score', fontsize=12)
        ax.set_title(f'{graph_name}\nOddball Score vs Degree\n(weights: distance, oddball: heavyvicinity W_i ~ E_i^β)', 
                    fontsize=14, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Statistics text
        stats_text = (
            f'Nodes: {len(df_merged)}\n'
            f'Degree range: [{df_merged["degree"].min()}, {df_merged["degree"].max()}]\n'
            f'Oddball range: [{df_merged["oddball_score"].min():.2f}, {df_merged["oddball_score"].max():.2f}]\n'
            f'Mean oddball: {df_merged["oddball_score"].mean():.2f}'
        )
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save plot
        output_file = output_dir / f"{graph_name}_oddball_vs_degree.png"
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
print(f"Total files found: {len(oddball_files)}")
print(f"Successfully processed: {processed}")
print(f"Skipped: {skipped}")
print(f"\nPlots saved to: {output_dir}")
print("\nDone!")
