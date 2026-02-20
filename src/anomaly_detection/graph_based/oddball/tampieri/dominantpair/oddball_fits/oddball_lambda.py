#!/usr/bin/env python
# coding: utf-8

# Oddball Algorithm for Weighted Networks - Eigenvalue vs Egonet Weight
# Relation: λ_i ~ W_i^beta
# λ_i = principal eigenvalue of adjacency matrix of egonet i
# W_i = total weight of egonet i
# beta = power law exponent (to be determined via regression)

import pickle
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh

# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi/weighted')
output_base = Path('/home/projects/safe/outputs/networks/oddball/weighted')
output_dir = output_base / 'ODB_3_scores'
plots_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_fits/lambda_fits')
output_dir.mkdir(parents=True, exist_ok=True)
plots_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print(f"Plots directory: {plots_dir}")

# Load all weighted visibility graphs
print("\n" + "="*60)
print("LOADING WEIGHTED VISIBILITY GRAPHS")
print("="*60)

graphs = {}

# Filter for p3 and p4, temp and rms only
target_patterns = ['p3_rms', 'p3_temp', 'p4_rms', 'p4_temp']

# Find all pickle files (excluding collection file)
pickle_files = sorted([f for f in input_dir.glob('G_NVG_*.pkl') if 'collection' not in f.name])
print(f"Found {len(pickle_files)} pickle files")
print(f"Filtering for: {target_patterns}")

for pkl_file in pickle_files:
    # Check if file matches any target pattern
    if not any(pattern in pkl_file.stem for pattern in target_patterns):
        continue
    
    try:
        with open(pkl_file, 'rb') as f:
            graph = pickle.load(f)
        
        # Extract graph name from filename
        graph_name = pkl_file.stem
        graphs[graph_name] = graph
        
        print(f"Loaded {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"Error loading {pkl_file.name}: {e}")

print(f"\nSuccessfully loaded {len(graphs)} graphs (filtered for p3/p4, temp/rms)")

def calculate_principal_eigenvalue(graph):
    """
    Calculate the principal (largest) eigenvalue of the adjacency matrix.
    """
    if graph.number_of_nodes() == 0:
        return 0.0
    
    if graph.number_of_nodes() == 1:
        return 0.0
    
    try:
        # Get adjacency matrix (weighted)
        adj_matrix = nx.adjacency_matrix(graph, weight='weight')
        
        # For small graphs, use dense eigenvalue computation
        if graph.number_of_nodes() < 100:
            adj_dense = adj_matrix.toarray()
            eigenvalues = eigh(adj_dense, eigvals_only=True)
            principal_eigenvalue = np.max(np.abs(eigenvalues))
        else:
            # For larger graphs, use sparse computation (only largest eigenvalue)
            eigenvalues = eigsh(adj_matrix, k=1, which='LM', return_eigenvectors=False)
            principal_eigenvalue = np.abs(eigenvalues[0])
        
        return principal_eigenvalue
    
    except Exception as e:
        print(f"    Warning: Could not compute eigenvalue: {e}")
        return 0.0

def calculate_egonet_lambda_weight_features(graph, node):
    """
    Calculate weighted egonet features for a given node.
    
    Returns:
        principal_eigenvalue (float): Principal eigenvalue of egonet adjacency matrix
        total_weight (float): Sum of all edge weights in the egonet
    """
    # Get ego network (node + its neighbors)
    ego_nodes = set([node] + list(graph.neighbors(node)))
    
    # Create subgraph of ego network
    egonet = graph.subgraph(ego_nodes)
    
    # Calculate principal eigenvalue
    principal_eigenvalue = calculate_principal_eigenvalue(egonet)
    
    # Calculate total weight (sum of all edge weights in egonet)
    total_weight = 0.0
    for u, v, data in egonet.edges(data=True):
        weight = data.get('weight', 1.0)  # Default to 1.0 if no weight
        total_weight += weight
    
    return principal_eigenvalue, total_weight

def oddball_lambda_analysis(graph, graph_name):
    """
    Apply Oddball algorithm using eigenvalue-weight relationship.
    Uses power law: λ_i ~ W_i^beta
    
    Returns:
        df: DataFrame with node features and oddball scores
        beta: Fitted power law exponent
        intercept: Regression intercept
    """
    
    print(f"\n{'='*60}")
    print(f"Analyzing: {graph_name}")
    print(f"{'='*60}")
    
    # Calculate egonet features for all nodes
    node_features = []
    
    for i, node in enumerate(graph.nodes()):
        if (i + 1) % 10000 == 0:
            print(f"  Processing node {i+1}/{graph.number_of_nodes()}...")
        
        principal_eigenvalue, total_weight = calculate_egonet_lambda_weight_features(graph, node)
        
        node_features.append({
            'node': node,
            'egonet_eigenvalue': principal_eigenvalue,
            'egonet_total_weight': total_weight,
            'degree': graph.degree(node)
        })
    
    # Create DataFrame
    df = pd.DataFrame(node_features)
    
    print(f"\nCalculated features for {len(df)} nodes")
    print(f"Eigenvalue range: [{df['egonet_eigenvalue'].min():.4f}, {df['egonet_eigenvalue'].max():.4f}]")
    print(f"Total weight range: [{df['egonet_total_weight'].min():.4f}, {df['egonet_total_weight'].max():.4f}]")
    
    # Log transform for power law: log(λ_i) ~ beta * log(W_i)
    # Filter out zero or negative values before log transform
    valid_mask = (df['egonet_eigenvalue'] > 0) & (df['egonet_total_weight'] > 0)
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) < 10:
        print(f"Warning: Not enough valid data points ({len(df_valid)}) for regression")
        df['log_eigenvalue'] = 0
        df['log_weight'] = 0
        df['predicted_log_eigenvalue'] = 0
        df['residual'] = 0
        df['oddball_score'] = 0
        return df, None, None
    
    df_valid['log_eigenvalue'] = np.log(df_valid['egonet_eigenvalue'])
    df_valid['log_weight'] = np.log(df_valid['egonet_total_weight'])
    
    # Fit linear regression in log-log space: log(λ) = beta * log(W) + intercept
    X = df_valid[['log_weight']].values
    y = df_valid['log_eigenvalue'].values
    
    # Remove any infinite or NaN values (extra safety)
    finite_mask = np.isfinite(X.flatten()) & np.isfinite(y)
    X_clean = X[finite_mask]
    y_clean = y[finite_mask]
    
    print(f"Valid data points for regression: {len(X_clean)}")
    
    if len(X_clean) < 10:
        print(f"Warning: Not enough clean data points for regression")
        df['log_eigenvalue'] = 0
        df['log_weight'] = 0
        df['predicted_log_eigenvalue'] = 0
        df['residual'] = 0
        df['oddball_score'] = 0
        return df, None, None
    
    # Fit regression
    reg = LinearRegression()
    reg.fit(X_clean, y_clean)
    
    beta = reg.coef_[0]
    intercept = reg.intercept_
    
    print(f"\nPower law fit: log(λ_i) = {beta:.4f} * log(W_i) + {intercept:.4f}")
    print(f"Beta (exponent): {beta:.4f}")
    print(f"Intercept: {intercept:.4f}")
    
    # Predict and calculate residuals for all valid points
    df_valid['predicted_log_eigenvalue'] = reg.predict(df_valid[['log_weight']].values)
    df_valid['residual'] = df_valid['log_eigenvalue'] - df_valid['predicted_log_eigenvalue']
    
    # Calculate oddball score (absolute residual / std of residuals)
    residual_std = df_valid['residual'].std()
    if residual_std > 0:
        df_valid['oddball_score'] = np.abs(df_valid['residual']) / residual_std
    else:
        df_valid['oddball_score'] = 0
    
    # Add beta and intercept to dataframe
    df_valid['beta'] = beta
    df_valid['intercept'] = intercept
    
    # Merge back to original dataframe
    df = df.merge(
        df_valid[['node', 'log_eigenvalue', 'log_weight', 'predicted_log_eigenvalue', 'residual', 'oddball_score', 'beta', 'intercept']],
        on='node',
        how='left'
    )
    
    # Fill NaN values for invalid nodes
    df.fillna({
        'log_eigenvalue': 0,
        'log_weight': 0,
        'predicted_log_eigenvalue': 0,
        'residual': 0,
        'oddball_score': 0,
        'beta': beta if beta is not None else 0,
        'intercept': intercept if intercept is not None else 0
    }, inplace=True)
    
    print(f"\nOddball score statistics:")
    print(f"  Mean: {df['oddball_score'].mean():.4f}")
    print(f"  Std: {df['oddball_score'].std():.4f}")
    print(f"  Min: {df['oddball_score'].min():.4f}")
    print(f"  Max: {df['oddball_score'].max():.4f}")
    print(f"  Median: {df['oddball_score'].median():.4f}")
    print(f"  95th percentile: {df['oddball_score'].quantile(0.95):.4f}")
    
    # Top 10 anomalous nodes
    top_outliers = df.nlargest(10, 'oddball_score')
    print(f"\nTop 10 anomalous nodes:")
    for idx, row in top_outliers.iterrows():
        print(f"  Node {int(row['node'])}: oddball_score={row['oddball_score']:.4f}, "
              f"λ_i={row['egonet_eigenvalue']:.4f}, W_i={row['egonet_total_weight']:.4f}, "
              f"residual={row['residual']:.4f}")
    
    return df, beta, intercept

# Process all graphs
print("\n" + "="*60)
print("RUNNING ODDBALL LAMBDA ANALYSIS")
print("="*60)

results_summary = []

for graph_name, graph in graphs.items():
    try:
        # Run oddball analysis
        df_result, beta, intercept = oddball_lambda_analysis(graph, graph_name)
        
        # Save results
        output_file = output_dir / f"{graph_name}_oddball_lambda_results.csv"
        df_result.to_csv(output_file, index=False)
        print(f"\n✓ Saved results: {output_file.name}")
        
        # Store summary
        results_summary.append({
            'graph_name': graph_name,
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'beta': beta,
            'intercept': intercept,
            'mean_oddball_score': df_result['oddball_score'].mean(),
            'max_oddball_score': df_result['oddball_score'].max()
        })
        
        # Create scatter plot: log(λ_i) vs log(W_i)
        if beta is not None and intercept is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Filter valid points for plotting
            plot_df = df_result[(df_result['log_eigenvalue'] > 0) & (df_result['log_weight'] > 0)].copy()
            
            # Scatter plot with color based on oddball score
            scatter = ax.scatter(
                plot_df['log_weight'],
                plot_df['log_eigenvalue'],
                c=plot_df['oddball_score'],
                cmap='viridis',
                alpha=0.5,
                s=10,
                edgecolors='none'
            )
            
            # Plot regression line
            x_line = np.array([plot_df['log_weight'].min(), plot_df['log_weight'].max()])
            y_line = beta * x_line + intercept
            ax.plot(x_line, y_line, 'r--', linewidth=2, 
                   label=f'Fit: y = {beta:.4f}x + {intercept:.4f}')
            
            # Labels and title
            ax.set_xlabel('log(W_i) - log(Egonet Total Weight)', fontsize=12)
            ax.set_ylabel('log(λ_i) - log(Principal Eigenvalue)', fontsize=12)
            ax.set_title(f'{graph_name} (weights: distance)\nEigenvalue vs Egonet Weight: log(λ_i) = β·log(W_i) + intercept\nβ={beta:.4f}, intercept={intercept:.4f}',
                        fontsize=13, fontweight='bold')
            
            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Oddball Score', fontsize=12)
            
            # Legend
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Save plot
            plot_file = plots_dir / f"{graph_name}_lambda_weight_fit.png"
            plt.tight_layout()
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Saved plot: {plot_file.name}")
        
    except Exception as e:
        print(f"\n✗ Error processing {graph_name}: {e}")
        import traceback
        traceback.print_exc()

# Save summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

summary_df = pd.DataFrame(results_summary)
summary_file = output_dir / 'oddball_lambda_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"\n✓ Saved summary: {summary_file}")

print(f"\nProcessed {len(results_summary)} graphs")
print(f"Results saved in: {output_dir}")
print(f"Plots saved in: {plots_dir}")

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
