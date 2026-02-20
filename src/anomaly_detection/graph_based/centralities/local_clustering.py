#!/usr/bin/env python
# coding: utf-8

# Calculate Local Clustering Coefficient for Visibility Graphs

import pickle
import networkx as nx
import pandas as pd
from pathlib import Path

# Paths
graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/centralities/local_clustering')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading visibility graphs...")
print(f"Input directory: {graphs_dir}")
print(f"Output directory: {output_dir}")
print("=" * 60)

# Load all graphs
graphs = {}
pickle_files = list(graphs_dir.glob('*.pkl'))

for pkl_file in pickle_files:
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both single graph and dictionary of graphs
    if isinstance(data, dict):
        # If it's a dict, check if it contains graphs or is a collection
        if 'all_nvg_graphs' in data or 'all_hvg_graphs' in data:
            # It's a collection file, skip it
            print(f"Skipping collection file: {pkl_file.name}")
            continue
        else:
            # It's a dict of graphs, extract them
            for name, graph in data.items():
                if hasattr(graph, 'number_of_nodes'):
                    graphs[name] = graph
                    print(f"Loaded {name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    elif hasattr(data, 'number_of_nodes'):
        # It's a single graph
        graph_name = pkl_file.stem
        graphs[graph_name] = data
        print(f"Loaded {graph_name}: {data.number_of_nodes()} nodes, {data.number_of_edges()} edges")

print(f"\nTotal graphs loaded: {len(graphs)}")

# Calculate local clustering coefficient for each graph
print("\n" + "=" * 60)
print("CALCULATING LOCAL CLUSTERING COEFFICIENT")
print("=" * 60)

clustering_results = {}

for graph_name, graph in graphs.items():
    print(f"\nProcessing {graph_name}...")
    
    try:
        # Calculate local clustering coefficient
        clustering = nx.clustering(graph)
        
        # Create DataFrame preserving node indices
        df_clustering = pd.DataFrame({
            'node': list(clustering.keys()),
            'local_clustering_coefficient': list(clustering.values())
        })
        
        # Sort by node index to preserve order
        df_clustering = df_clustering.sort_values('node').reset_index(drop=True)
        
        # Store results
        clustering_results[graph_name] = df_clustering
        
        # Save individual CSV
        csv_file = output_dir / f"{graph_name}_clustering.csv"
        df_clustering.to_csv(csv_file, index=False)
        
        print(f"  Nodes: {len(clustering)}")
        print(f"  Max clustering: {df_clustering['local_clustering_coefficient'].max():.6f}")
        print(f"  Mean clustering: {df_clustering['local_clustering_coefficient'].mean():.6f}")
        print(f"  Saved: {csv_file.name}")
        
    except Exception as e:
        print(f"  Error calculating clustering for {graph_name}: {e}")
        continue

# Save all results as pickle
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results_pickle = output_dir / "clustering_results.pkl"
with open(results_pickle, 'wb') as f:
    pickle.dump(clustering_results, f)
print(f"All results saved: {results_pickle}")

# Create summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

summary_data = []
for graph_name, df in clustering_results.items():
    summary_data.append({
        'graph': graph_name,
        'num_nodes': len(df),
        'max_clustering': df['local_clustering_coefficient'].max(),
        'mean_clustering': df['local_clustering_coefficient'].mean(),
        'median_clustering': df['local_clustering_coefficient'].median(),
        'std_clustering': df['local_clustering_coefficient'].std()
    })

summary_df = pd.DataFrame(summary_data)
summary_file = output_dir / "clustering_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"\nSummary statistics:")
print(summary_df.to_string(index=False))
print(f"\nSummary saved: {summary_file}")

print("\n" + "=" * 60)
print("LOCAL CLUSTERING COEFFICIENT CALCULATION COMPLETED")
print("=" * 60)
print(f"Graphs processed: {len(clustering_results)}")
print(f"Output directory: {output_dir}")
print("Done!")
