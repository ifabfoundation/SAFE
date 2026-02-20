# Calculate Degree Distribution for Bonfiglioli Graphs

import pickle
import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs')
output_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs/degrees')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading Bonfiglioli graphs...")
print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")
print("=" * 60)

# Load all graphs
graphs = {}
pickle_files = [f for f in input_dir.glob('*.pkl') if 'collection' not in f.name]

for pkl_file in pickle_files:
    try:
        with open(pkl_file, 'rb') as f:
            graph = pickle.load(f)
        
        graph_name = pkl_file.stem
        graphs[graph_name] = graph
        
        print(f"Loaded {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"Error loading {pkl_file.name}: {e}")

print(f"\nSuccessfully loaded {len(graphs)} graphs")

# Calculate degree distribution for each graph
print("\n" + "=" * 60)
print("CALCULATING DEGREE DISTRIBUTIONS")
print("=" * 60)

for graph_name, graph in graphs.items():
    print(f"\nProcessing {graph_name}...")
    
    try:
        # Get degree for each node
        degrees = dict(graph.degree())
        
        # Create DataFrame
        df_degrees = pd.DataFrame({
            'node': list(degrees.keys()),
            'degree': list(degrees.values())
        })
        
        # Sort by node index to preserve order
        df_degrees = df_degrees.sort_values('node').reset_index(drop=True)
        
        # Save individual CSV
        csv_file = output_dir / f"{graph_name}_degrees.csv"
        df_degrees.to_csv(csv_file, index=False)
        
        # Print statistics
        print(f"  Nodes: {len(degrees)}")
        print(f"  Min degree: {df_degrees['degree'].min()}")
        print(f"  Max degree: {df_degrees['degree'].max()}")
        print(f"  Mean degree: {df_degrees['degree'].mean():.2f}")
        print(f"  Median degree: {df_degrees['degree'].median():.0f}")
        print(f"  Saved: {csv_file.name}")
        
    except Exception as e:
        print(f"  Error calculating degrees for {graph_name}: {e}")
        continue

print("\n" + "=" * 60)
print("DEGREE CALCULATION COMPLETED")
print("=" * 60)
print(f"Graphs processed: {len(graphs)}")
print(f"Output directory: {output_dir}")
print("Done!")
