# Plot Oddball Score vs Node Degree
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores')
graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_vs_degrees')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading oddball results from: {oddball_dir}")
print(f"Loading graphs from: {graphs_dir}")
print(f"Output directory: {output_dir}")

# Load oddball results
oddball_pickle = oddball_dir / "oddball_results.pkl"
print(f"\nLoading {oddball_pickle}...")
with open(oddball_pickle, 'rb') as f:
    oddball_results = pickle.load(f)

print(f"Loaded oddball results for {len(oddball_results)} graphs")

# Process each graph
for graph_name, oddball_df in oddball_results.items():
    try:
        # Load corresponding graph to get degrees
        graph_file = graphs_dir / f"{graph_name}.pkl"
        
        if not graph_file.exists():
            print(f"Graph file not found: {graph_file.name}")
            continue
            
        with open(graph_file, 'rb') as f:
            graph = pickle.load(f)
        
        # Calculate degrees for all nodes
        degrees = dict(graph.degree())
        
        # Add degrees to oddball dataframe
        oddball_df['degree'] = oddball_df['node'].map(degrees)
        
        # Remove rows with missing degrees
        plot_df = oddball_df.dropna(subset=['degree', 'oddball_score'])
        
        if len(plot_df) == 0:
            print(f"No valid data for {graph_name}")
            continue
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_df['degree'], plot_df['oddball_score'], 
                   alpha=0.6, s=30, color='blue')
        
        plt.xlabel('Node Degree')
        plt.ylabel('Oddball Score')
        plt.title(f'Oddball Score vs Degree - {graph_name}')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / f"{graph_name}_oddball_vs_degree.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_file.name}")
        
    except Exception as e:
        print(f"Error processing {graph_name}: {e}")
        continue

print(f"\nPlots created: {len(list(output_dir.glob('*.png')))} files")
print(f"Output directory: {output_dir}")
print("Done!")
