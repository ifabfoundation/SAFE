#!/usr/bin/env python
# coding: utf-8

# Analyze secondary branches in oddball weighted results
# Identifies multiple power law relationships in log(W) vs log(E) plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# Paths
results_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores/weighted_scores')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_fits/weighted_fits_with_branches')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYZING SECONDARY BRANCHES IN ODDBALL WEIGHTED RESULTS")
print("=" * 80)

def identify_branches(df, main_beta, main_intercept, log_e_threshold=6.0, n_branches=4):
    """
    Identify secondary branches using residual-based clustering with independent fitting.
    Clusters on residuals to find parallel branches, then re-fits each to get actual beta.
    
    Parameters:
        df: DataFrame with log_edges, log_weight columns
        main_beta: Beta from main fit
        main_intercept: Intercept from main fit
        log_e_threshold: Only consider points with log(E) > this value
        n_branches: Expected number of branches (including main)
    
    Returns:
        List of branch info dicts
    """
    # Calculate residuals from main fit
    df['main_predicted'] = main_beta * df['log_edges'] + main_intercept
    df['residual'] = df['log_weight'] - df['main_predicted']
    
    # Filter points beyond threshold where branches are visible
    df_filtered = df[df['log_edges'] > log_e_threshold].copy()
    
    print(f"  Points with log(E) > {log_e_threshold}: {len(df_filtered)} / {len(df)}")
    
    if len(df_filtered) < 1000:
        print(f"  Not enough filtered points to identify branches")
        return []
    
    # Step 1: Cluster on residuals to identify parallel branches
    residuals = df_filtered['residual'].values.reshape(-1, 1)
    
    print(f"  Using KMeans clustering on residuals with {n_branches} clusters...")
    kmeans = KMeans(n_clusters=n_branches, random_state=42, n_init=20, max_iter=300)
    df_filtered['cluster'] = kmeans.fit_predict(residuals)
    
    # Get unique clusters
    unique_clusters = sorted(df_filtered['cluster'].unique())
    
    print(f"  Final clusters: {len(unique_clusters)}")
    
    branches_info = []
    
    for cluster_id in unique_clusters:
        cluster_points = df_filtered[df_filtered['cluster'] == cluster_id].copy()
        
        if len(cluster_points) < 100:  # Skip small clusters
            continue
        
        # Fit regression for this cluster
        X_cluster = cluster_points[['log_edges']].values
        y_cluster = cluster_points['log_weight'].values
        
        reg = LinearRegression()
        reg.fit(X_cluster, y_cluster)
        
        branch_beta = reg.coef_[0]
        branch_intercept = reg.intercept_
        
        # Calculate R² for this branch
        y_pred = reg.predict(X_cluster)
        ss_res = np.sum((y_cluster - y_pred)**2)
        ss_tot = np.sum((y_cluster - y_cluster.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate mean residual for this branch
        mean_residual = cluster_points['residual'].mean()
        
        branches_info.append({
            'cluster_id': cluster_id,
            'beta': branch_beta,
            'intercept': branch_intercept,
            'n_points': len(cluster_points),
            'log_edges_range': (cluster_points['log_edges'].min(), cluster_points['log_edges'].max()),
            'mean_residual': mean_residual,
            'r2': r2,
            'nodes': cluster_points.index.tolist()  # Store node indices
        })
        
        print(f"    Branch {cluster_id}: β={branch_beta:.4f}, int={branch_intercept:.4f}, "
              f"R²={r2:.4f}, pts={len(cluster_points)}, mean_res={mean_residual:.3f}")
    
    # Sort branches by mean residual (bottom to top)
    branches_info.sort(key=lambda x: x['mean_residual'])
    
    return branches_info

# Find all oddball result files
result_files = list(results_dir.glob('*_oddball_weighted_results.csv'))

# Filter for G_NVG_temp_mot_marcia only (for testing)
result_files = [f for f in result_files if 'G_NVG_temp_mot_marcia' in f.stem]

print(f"\nFound {len(result_files)} result files (filtered for G_NVG_temp_mot_marcia)")

# Process each graph
for result_file in result_files:
    graph_name = result_file.stem.replace('_oddball_weighted_results', '')
    print(f"\n{'='*80}")
    print(f"Processing: {graph_name}")
    print(f"{'='*80}")
    
    try:
        # Load results
        df = pd.read_csv(result_file)
        
        # Check required columns
        required_cols = ['log_edges', 'log_weight', 'beta', 'intercept', 'oddball_score']
        if not all(col in df.columns for col in required_cols):
            print(f"  Missing required columns, skipping...")
            continue
        
        # Filter valid data
        df_valid = df[(df['log_edges'] > 0) & (df['log_weight'] > 0)].copy()
        
        if len(df_valid) < 100:
            print(f"  Not enough valid data points ({len(df_valid)}), skipping...")
            continue
        
        # Get main fit parameters
        main_beta = df_valid['beta'].iloc[0]
        main_intercept = df_valid['intercept'].iloc[0]
        
        print(f"  Main fit: β={main_beta:.4f}, intercept={main_intercept:.4f}")
        print(f"  Valid data points: {len(df_valid)}")
        
        # Identify branches (reduced to 3 to exclude less relevant branch)
        branches = identify_branches(df_valid, main_beta, main_intercept, log_e_threshold=6.0, n_branches=3)
        
        # Create single plot: Power law with branches
        fig, ax1 = plt.subplots(figsize=(10, 8))
        
        # Scatter plot with color based on oddball score
        scatter1 = ax1.scatter(
            df_valid['log_edges'],
            df_valid['log_weight'],
            c=df_valid['oddball_score'],
            cmap='viridis',
            alpha=0.5,
            s=10,
            edgecolors='none'
        )
        
        # Plot main regression line
        x_line = np.array([df_valid['log_edges'].min(), df_valid['log_edges'].max()])
        y_line = main_beta * x_line + main_intercept
        ax1.plot(x_line, y_line, 'r--', linewidth=2.5, 
               label=f'Main: β={main_beta:.4f}, int={main_intercept:.4f}')
        
        # Plot secondary branches
        colors = ['orange', 'cyan', 'magenta', 'lime', 'yellow']
        for i, branch in enumerate(branches):
            x_branch = np.array([branch['log_edges_range'][0], branch['log_edges_range'][1]])
            y_branch = branch['beta'] * x_branch + branch['intercept']
            color = colors[i % len(colors)]
            ax1.plot(x_branch, y_branch, '--', linewidth=2.5, color=color,
                   label=f"Branch {i+1}: β={branch['beta']:.4f}, int={branch['intercept']:.4f}")
        
        # Labels and title
        title_text = f'{graph_name}\nEgonet Weight Power Law Analysis'
        if branches:
            title_text += f'\nMain + {len(branches)} branch(es)'
        
        ax1.set_xlabel('log(E_i) - log(Egonet Edges)', fontsize=11)
        ax1.set_ylabel('log(W_i) - log(Egonet Total Weight)', fontsize=11)
        ax1.set_title(title_text, fontsize=12, fontweight='bold')
        
        # Colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Oddball Score', fontsize=10)
        
        # Legend
        ax1.legend(fontsize=9, loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / f"{graph_name}_branches_analysis.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: {plot_file.name}")
        
        # Save branch info to CSV
        if branches:
            # Save summary
            branches_summary = []
            for b in branches:
                branches_summary.append({
                    'cluster_id': b['cluster_id'],
                    'beta': b['beta'],
                    'intercept': b['intercept'],
                    'n_points': b['n_points'],
                    'mean_residual': b['mean_residual'],
                    'r2': b['r2']
                })
            branches_df = pd.DataFrame(branches_summary)
            branches_df['graph_name'] = graph_name
            branches_csv = output_dir / f"{graph_name}_branches_info.csv"
            branches_df.to_csv(branches_csv, index=False)
            print(f"  ✓ Saved branch info: {branches_csv.name}")
            
            # Save nodes for each branch
            for i, branch in enumerate(branches):
                nodes_df = pd.DataFrame({
                    'node': branch['nodes'],
                    'branch_id': i,
                    'branch_beta': branch['beta'],
                    'branch_intercept': branch['intercept']
                })
                nodes_csv = output_dir / f"{graph_name}_branch_{i}_nodes.csv"
                nodes_df.to_csv(nodes_csv, index=False)
                print(f"  ✓ Saved branch {i} nodes ({len(branch['nodes'])} nodes): {nodes_csv.name}")
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory: {output_dir}")
