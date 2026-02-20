# Oddball Algorithm for Anomaly Detection in Networks
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_base = Path('/home/projects/safe/outputs/networks/oddball')
output_dir = output_base / 'oddball_scores'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Load all visibility graphs
print("\nLoading visibility graphs...")
graphs = {}

# Target graphs to analyze
target_graphs = ['G_NVG_p3_rms', 'G_NVG_p3_temp', 'G_NVG_p4_rms', 'G_NVG_p4_temp']

# Find all pickle files
pickle_files = list(input_dir.glob('*.pkl'))
print(f"Found {len(pickle_files)} pickle files")

for pkl_file in pickle_files:
    try:
        # Extract graph name from filename
        graph_name = pkl_file.stem
        
        # Skip if not in target list
        if graph_name not in target_graphs:
            continue
        
        with open(pkl_file, 'rb') as f:
            graph = pickle.load(f)
        
        graphs[graph_name] = graph
        
        print(f"Loaded {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"Error loading {pkl_file.name}: {e}")

print(f"\nSuccessfully loaded {len(graphs)} graphs (target: {', '.join(target_graphs)})")

def calculate_egonet_features(graph, node):
    """
    Calculate egonet features for a given node.
    Returns: (num_nodes, num_edges)
    """
    # Get ego network (node + its neighbors)
    ego_nodes = set([node] + list(graph.neighbors(node)))
    
    # Create subgraph of ego network
    egonet = graph.subgraph(ego_nodes)
    
    # Count nodes and edges in egonet
    num_nodes = egonet.number_of_nodes()
    num_edges = egonet.number_of_edges()
    
    return num_nodes, num_edges

def oddball_analysis(graph, graph_name):
    """
    Apply Oddball algorithm to detect anomalous nodes.
    Returns dataframe with node features and oddball scores.
    """
    
    # Calculate egonet features for all nodes
    node_features = []
    
    for node in graph.nodes():
        num_nodes, num_edges = calculate_egonet_features(graph, node)
        
        node_features.append({
            'node': node,
            'egonet_nodes': num_nodes,
            'egonet_edges': num_edges,
            'degree': graph.degree(node)
        })
    
    # Create DataFrame
    df = pd.DataFrame(node_features)
    
    # Log transform for better linear relationship
    df['log_nodes'] = np.log1p(df['egonet_nodes'])  # log(1 + x)
    df['log_edges'] = np.log1p(df['egonet_edges'])
    
    # Fit linear regression: log(edges) ~ log(nodes)
    X = df[['log_nodes']].values
    y = df['log_edges'].values
    
    # Remove any infinite or NaN values
    valid_mask = np.isfinite(X.flatten()) & np.isfinite(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    
    if len(X_clean) < 2:
        print(f"Not enough valid data points for regression")
        df['predicted_log_edges'] = 0
        df['residual'] = 0
        df['oddball_score'] = 0
        return df
    
    # Fit regression
    reg = LinearRegression()
    reg.fit(X_clean, y_clean)
    
    # Predict for all nodes
    df['predicted_log_edges'] = reg.predict(X)
    
    # Calculate residuals (actual - predicted)
    df['residual'] = df['log_edges'] - df['predicted_log_edges']
    
    # Oddball score is the absolute residual
    df['oddball_score'] = np.abs(df['residual'])
    
    # Add alpha (power law exponent)
    df['alpha'] = reg.coef_[0] if len(reg.coef_) > 0 else 0
    
    # Rank nodes by oddball score
    df['oddball_rank'] = df['oddball_score'].rank(ascending=False)
    
    print(f"Alpha = {df['alpha'].iloc[0]:.4f}")
    
    return df

# Apply Oddball analysis to all graphs
print("\nApplying Oddball analysis to all graphs...")

oddball_results = {}

for graph_name, graph in graphs.items():
    try:
        # Apply oddball analysis
        result_df = oddball_analysis(graph, graph_name)
        oddball_results[graph_name] = result_df
        
        # Save individual results
        csv_file = output_dir / f"{graph_name}_oddball_results.csv"
        result_df.to_csv(csv_file, index=False)
        print(f"Results saved: {csv_file.name}")
        
    except Exception as e:
        print(f"Error processing {graph_name}: {e}")
        continue

# Save all results as pickle
results_pickle = output_dir / "oddball_results.pkl"
with open(results_pickle, 'wb') as f:
    pickle.dump(oddball_results, f)
print(f"All results saved: {results_pickle.name}")

# Create oddball fit plots (observed vs predicted)
print("\nCreating oddball fit plots...")
fit_plots_dir = output_base / "oddball_fits"
fit_plots_dir.mkdir(parents=True, exist_ok=True)

for graph_name, oddball_df in oddball_results.items():
    try:
        # Filter out invalid data
        valid_data = oddball_df.dropna(subset=['log_nodes', 'log_edges', 'predicted_log_edges'])
        
        if len(valid_data) == 0:
            print(f"No valid data for {graph_name}")
            continue
            
        # Create single plot
        plt.figure(figsize=(10, 8))
        
        # Plot observed data points colored by oddball score
        scatter = plt.scatter(valid_data['log_nodes'], valid_data['log_edges'], 
                   c=valid_data['oddball_score'], cmap='viridis',
                   alpha=0.6, s=30, edgecolors='none')
        
        # Plot regression line (predicted values)
        x_range = np.linspace(valid_data['log_nodes'].min(), valid_data['log_nodes'].max(), 100)
        if len(valid_data) > 0:
            alpha = valid_data['alpha'].iloc[0]
            # Calculate intercept from predicted values
            intercept = valid_data['predicted_log_edges'].iloc[0] - alpha * valid_data['log_nodes'].iloc[0]
            y_pred_line = alpha * x_range + intercept
            plt.plot(x_range, y_pred_line, 'r--', linewidth=2, 
                    label=f'Fit: y = {alpha:.4f}x + {intercept:.4f}')
        
        plt.xlabel('Log(Egonet Nodes)', fontsize=12)
        plt.ylabel('Log(Egonet Edges)', fontsize=12)
        plt.title(f'Oddball cliquestar - {graph_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add colorbar for oddball score
        cbar = plt.colorbar(scatter)
        cbar.set_label('Oddball Score', fontsize=10)
        
        # Add statistics
        if len(valid_data) > 0:
            alpha = valid_data['alpha'].iloc[0]
            intercept = valid_data['predicted_log_edges'].iloc[0] - alpha * valid_data['log_nodes'].iloc[0]
            
            stats_text = f"""α = {alpha:.4f}
intercept = {intercept:.4f}
Points: {len(valid_data)}"""
            
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        fit_plot_file = fit_plots_dir / f"{graph_name}_oddball_fit.png"
        plt.savefig(fit_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Fit plot saved: {fit_plot_file.name}")
        
    except Exception as e:
        print(f"Error creating fit plot for {graph_name}: {e}")
        continue

print(f"\nODDBALL ANALYSIS COMPLETED")
print(f"Graphs analyzed: {len(oddball_results)}/{len(graphs)}")
print(f"Fit plots created: {len(list(fit_plots_dir.glob('*.png')))} files")
print(f"Output directory: {output_dir}")
print("Results saved successfully!")

# Special analysis for G_NVG_p3_temp: remove top 2 oddball nodes and recompute
print(f"\n{'='*60}")
print(f"SPECIAL ANALYSIS: G_NVG_p3_temp without top 2 oddball nodes")
print(f"{'='*60}")

if 'G_NVG_p3_temp' in oddball_results and 'G_NVG_p3_temp' in graphs:
    # Get top 2 oddball nodes
    df_p3_temp = oddball_results['G_NVG_p3_temp']
    top_2_nodes = df_p3_temp.nlargest(2, 'oddball_score')['node'].values
    
    print(f"\nTop 2 oddball nodes to remove:")
    for node in top_2_nodes:
        node_data = df_p3_temp[df_p3_temp['node'] == node].iloc[0]
        print(f"  Node {int(node)}: oddball_score = {node_data['oddball_score']:.6f}, degree = {int(node_data['degree'])}")
    
    # Create new graph without these nodes
    original_graph = graphs['G_NVG_p3_temp']
    filtered_graph = original_graph.copy()
    filtered_graph.remove_nodes_from(top_2_nodes)
    
    print(f"\nOriginal graph: {original_graph.number_of_nodes()} nodes, {original_graph.number_of_edges()} edges")
    print(f"Filtered graph: {filtered_graph.number_of_nodes()} nodes, {filtered_graph.number_of_edges()} edges")
    
    # Recompute oddball for filtered graph
    print(f"\nRecomputing oddball analysis for filtered graph...")
    result_df_filtered = oddball_analysis(filtered_graph, 'G_NVG_p3_temp_filtered')
    
    # Save filtered results
    csv_file_filtered = output_dir / "G_NVG_p3_temp_filtered_oddball_results.csv"
    result_df_filtered.to_csv(csv_file_filtered, index=False)
    print(f"Filtered results saved: {csv_file_filtered.name}")
    
    # Create comparison plot
    print(f"\nCreating filtered fit plot...")
    
    valid_data_filtered = result_df_filtered.dropna(subset=['log_nodes', 'log_edges', 'predicted_log_edges'])
    
    if len(valid_data_filtered) > 0:
        plt.figure(figsize=(10, 8))
        
        # Plot observed data points colored by oddball score
        scatter = plt.scatter(valid_data_filtered['log_nodes'], valid_data_filtered['log_edges'], 
                   c=valid_data_filtered['oddball_score'], cmap='viridis',
                   alpha=0.6, s=30, edgecolors='none')
        
        # Plot regression line
        x_range = np.linspace(valid_data_filtered['log_nodes'].min(), valid_data_filtered['log_nodes'].max(), 100)
        alpha_filtered = valid_data_filtered['alpha'].iloc[0]
        intercept_filtered = valid_data_filtered['predicted_log_edges'].iloc[0] - alpha_filtered * valid_data_filtered['log_nodes'].iloc[0]
        y_pred_line = alpha_filtered * x_range + intercept_filtered
        plt.plot(x_range, y_pred_line, 'r--', linewidth=2, 
                label=f'Fit: y = {alpha_filtered:.4f}x + {intercept_filtered:.4f}')
        
        plt.xlabel('Log(Egonet Nodes)', fontsize=12)
        plt.ylabel('Log(Egonet Edges)', fontsize=12)
        plt.title(f'Oddball Regression Fit - G_NVG_p3_temp (Filtered - Top 2 Removed)', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Oddball Score', fontsize=10)
        
        # Add statistics
        stats_text = f"""α = {alpha_filtered:.4f}
intercept = {intercept_filtered:.4f}
Points: {len(valid_data_filtered)}
Removed nodes: {list(top_2_nodes)}"""
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round',
                facecolor='white', alpha=0.8), fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        fit_plot_file_filtered = fit_plots_dir / "G_NVG_p3_temp_filtered_oddball_fit.png"
        plt.savefig(fit_plot_file_filtered, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Filtered fit plot saved: {fit_plot_file_filtered.name}")
        
        # Print comparison statistics
        alpha_original = df_p3_temp['alpha'].iloc[0]
        print(f"\n{'='*60}")
        print(f"COMPARISON:")
        print(f"{'='*60}")
        print(f"Original α: {alpha_original:.4f}")
        print(f"Filtered α: {alpha_filtered:.4f}")
        print(f"Difference: {abs(alpha_original - alpha_filtered):.4f}")
        
else:
    print("G_NVG_p3_temp not found in results, skipping filtered analysis")

print(f"\n{'='*60}")
print(f"ALL ANALYSIS COMPLETED")
print(f"{'='*60}")
