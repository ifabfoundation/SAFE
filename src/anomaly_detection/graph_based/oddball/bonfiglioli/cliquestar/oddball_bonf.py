# Oddball Algorithm for Anomaly Detection in Bonfiglioli Networks

#cliquestar for bonfiglioli lab data
#E_i = N_i^alpha
#compute also fits with intercept

import pickle
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs')
output_base = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli')
output_dir = output_base / 'oddball_scores/cliquestar_scores'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Target columns for Bonfiglioli
target_keys = ['tan_oil', 'temp_oil_marcia', 'temp_pt100_oil', 'vib_mot_marcia_rms', 
               'vib_rid_marc2_rms', 'vib_rid_marcia_rms', 'temp_cassa_riduttore', 'temp_mot_marcia']

# Load all visibility graphs
print("\nLoading Bonfiglioli visibility graphs (NVG only, filtered by target keys)...")
graphs = {}

# Find all pickle files (NVG only, excluding collection file)
pickle_files = [f for f in input_dir.glob('G_NVG_*.pkl') if 'collection' not in f.name]
print(f"Found {len(pickle_files)} NVG pickle files")

for pkl_file in pickle_files:
    try:
        # Extract graph name from filename
        graph_name = pkl_file.stem
        
        # Check if graph name contains any of the target keys
        if not any(key.lower() in graph_name.lower() for key in target_keys):
            continue
        
        with open(pkl_file, 'rb') as f:
            graph = pickle.load(f)
        
        graphs[graph_name] = graph
        
        print(f"Loaded {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"Error loading {pkl_file.name}: {e}")

print(f"\nSuccessfully loaded {len(graphs)} graphs")

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
    
    # Add alpha (power law exponent) and intercept
    df['alpha'] = reg.coef_[0] if len(reg.coef_) > 0 else 0
    df['intercept'] = reg.intercept_
    
    # Rank nodes by oddball score
    df['oddball_rank'] = df['oddball_score'].rank(ascending=False)
    
    print(f"Alpha = {df['alpha'].iloc[0]:.4f}, Intercept = {df['intercept'].iloc[0]:.4f}")
    
    return df

# Apply Oddball analysis to all graphs
print("\n" + "="*60)
print("APPLYING ODDBALL ANALYSIS")
print("="*60)

oddball_results = {}

for graph_name, graph in graphs.items():
    print(f"\nProcessing {graph_name}...")
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
print(f"\nAll results saved: {results_pickle.name}")

# Create oddball fit plots (observed vs predicted)
print("\n" + "="*60)
print("CREATING ODDBALL FIT PLOTS")
print("="*60)

fit_plots_dir = output_base / "oddball_fits/cliquestar_fits"
fit_plots_dir.mkdir(parents=True, exist_ok=True)

for graph_name, oddball_df in oddball_results.items():
    try:
        # Filter out invalid data
        valid_data = oddball_df.dropna(subset=['log_nodes', 'log_edges', 'predicted_log_edges'])
        
        if len(valid_data) == 0:
            print(f"No valid data for {graph_name}")
            continue
            
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot observed data points with colorbar for oddball score
        scatter = ax.scatter(valid_data['log_nodes'], valid_data['log_edges'], 
                   c=valid_data['oddball_score'], cmap='viridis',
                   alpha=0.6, s=30, edgecolors='none')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Oddball Score', fontsize=11)
        
        # Plot regression line (predicted values)
        x_range = np.linspace(valid_data['log_nodes'].min(), valid_data['log_nodes'].max(), 100)
        if len(valid_data) > 0:
            alpha = valid_data['alpha'].iloc[0]
            intercept = valid_data['intercept'].iloc[0]
            y_pred_line = alpha * x_range + intercept
            ax.plot(x_range, y_pred_line, 'r--', linewidth=2, 
                   label=f'Fit: y = {alpha:.4f}x + {intercept:.4f}')
        
        ax.set_xlabel('Log(Egonet Nodes)', fontsize=12)
        ax.set_ylabel('Log(Egonet Edges)', fontsize=12)
        ax.set_title(f'Oddball cliquestar - {graph_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if len(valid_data) > 0:
            alpha = valid_data['alpha'].iloc[0]
            intercept = valid_data['intercept'].iloc[0]
            
            stats_text = f"""α = {alpha:.4f}
intercept = {intercept:.4f}
Points: {len(valid_data)}"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round',
                    facecolor='white', alpha=0.8), fontsize=10)
        
        # Save plot
        fit_plot_file = fit_plots_dir / f"{graph_name}_oddball_fit.png"
        plt.savefig(fit_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Fit plot saved: {fit_plot_file.name}")
        
    except Exception as e:
        print(f"Error creating fit plot for {graph_name}: {e}")
        continue

print("\n" + "="*60)
print("ODDBALL ANALYSIS COMPLETED")
print("="*60)
print(f"Target keys: {target_keys}")
print(f"Graphs analyzed: {len(oddball_results)} (NVG only)")
print(f"Fit plots created: {len(list(fit_plots_dir.glob('*.png')))} files")
print(f"Output directory: {output_dir}")
print("Results saved successfully!")
