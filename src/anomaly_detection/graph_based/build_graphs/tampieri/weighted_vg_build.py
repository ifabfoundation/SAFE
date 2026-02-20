#!/usr/bin/env python
# coding: utf-8

# Build Weighted Visibility Graphs with 'distance'

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from ts2vg import NaturalVG, HorizontalVG
import networkx as nx

# Load dataset
data_file = Path('/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv')

print(f"Loading dataset from: {data_file}")
df = pd.read_csv(data_file, index_col=0)

print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")

# Get all columns for processing
data_columns = df.columns.tolist()

# Define target graphs to build
target_graphs = ['G_NVG_p3_acc_rms', 'G_NVG_p3_temp', 'G_NVG_p4_acc_rms', 'G_NVG_p4_temp']

# Extract column names from target graphs
target_columns = []
for target in target_graphs:
    # Remove 'G_NVG_' prefix to get column name
    col_name = target.replace('G_NVG_', '')
    # Find matching column in dataframe
    matching_cols = [col for col in data_columns if col_name in col]
    if matching_cols:
        target_columns.append(matching_cols[0])
    else:
        print(f"Warning: No matching column found for {target}")

print(f"\n{'='*60}")
print(f"BUILDING WEIGHTED VISIBILITY GRAPHS")
print(f"{'='*60}")
print(f"Target graphs: {target_graphs}")
print(f"Target columns: {target_columns}")

# Output directory
output_dir = Path('/home/projects/safe/outputs/networks/grafi/weighted')
output_dir.mkdir(parents=True, exist_ok=True)

# Storage for all graphs
all_nvg_graphs = {}
all_hvg_graphs = {}

# Build visibility graphs for each target column
for col in target_columns:
    print(f"\n{'='*60}")
    print(f"Processing column: {col}")
    print(f"{'='*60}")
    
    # Get time series data
    ts_data = df[col].values
    print(f"Time series length: {len(ts_data)}")
    print(f"Min: {ts_data.min():.4f}, Max: {ts_data.max():.4f}, Mean: {ts_data.mean():.4f}")
    
    # Build Natural Visibility Graph (NVG) - Weighted
    print(f"\nBuilding Natural Visibility Graph (NVG) - weighted...")
    nvg = NaturalVG(weighted='distance')
    nvg.build(ts_data)
    G_nvg = nvg.as_networkx()

    
    graph_name_nvg = f"G_NVG_{col}"
    all_nvg_graphs[graph_name_nvg] = G_nvg
    
    print(f"  {graph_name_nvg}: {G_nvg.number_of_nodes()} nodes, {G_nvg.number_of_edges()} edges")
    
    # Build Horizontal Visibility Graph (HVG) - Weighted
    print(f"Building Horizontal Visibility Graph (HVG) - weighted...")
    hvg = HorizontalVG(weighted='distance')
    hvg.build(ts_data)
    G_hvg = hvg.as_networkx()
    
    
    graph_name_hvg = f"G_HVG_{col}"
    all_hvg_graphs[graph_name_hvg] = G_hvg
    
    print(f"  {graph_name_hvg}: {G_hvg.number_of_nodes()} nodes, {G_hvg.number_of_edges()} edges")

# Save all graphs
print(f"\n{'='*60}")
print(f"SAVING GRAPHS")
print(f"{'='*60}")

# Save individual graph files
for graph_name, graph in all_nvg_graphs.items():
    safe_graph_name = graph_name.replace('/', '_').replace('\\', '_')
    pkl_file = output_dir / f"{safe_graph_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved: {pkl_file.name}")

for graph_name, graph in all_hvg_graphs.items():
    safe_graph_name = graph_name.replace('/', '_').replace('\\', '_')
    pkl_file = output_dir / f"{safe_graph_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved: {pkl_file.name}")

# Save collection file
collection = {
    'all_nvg_graphs': all_nvg_graphs,
    'all_hvg_graphs': all_hvg_graphs
}
collection_file = output_dir / 'all_weighted_visibility_graphs_collection.pkl'
with open(collection_file, 'wb') as f:
    pickle.dump(collection, f)
print(f"\nSaved collection: {collection_file.name}")

# Create summary file with weight statistics
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")

summary_lines = []
summary_lines.append(f"Total NVG graphs: {len(all_nvg_graphs)}")
summary_lines.append(f"Total HVG graphs: {len(all_hvg_graphs)}")

for graph_name, graph in {**all_nvg_graphs, **all_hvg_graphs}.items():
    weights = [data['weight'] for u, v, data in graph.edges(data=True)]
    if weights:
        summary_lines.append(f"\n{graph_name}:")
        summary_lines.append(f"  Min weight: {min(weights):.6f}")
        summary_lines.append(f"  Max weight: {max(weights):.6f}")
        summary_lines.append(f"  Mean weight: {np.mean(weights):.6f}")
        summary_lines.append(f"  Median weight: {np.median(weights):.6f}")

summary_file = output_dir / 'weighted_graphs_summary.txt'
with open(summary_file, 'w') as f:
    f.write('\n'.join(summary_lines))

print('\n'.join(summary_lines))
print(f"\nSummary saved: {summary_file}")

# Now calculate degree distributions from saved graphs
print(f"\n{'='*60}")
print(f"CALCULATING DEGREE DISTRIBUTIONS")
print(f"{'='*60}")

# Directory for degree distributions
degrees_dir = Path('/home/projects/safe/outputs/networks/grafi/weighted/degrees')
degrees_dir.mkdir(parents=True, exist_ok=True)

# Get list of saved graph files
graph_files = list(output_dir.glob('G_*.pkl'))
print(f"Found {len(graph_files)} saved graph files")

# Calculate and save degree distribution for each saved graph
for pkl_file in graph_files:
    print(f"\nLoading and processing: {pkl_file.name}")
    
    # Load graph from disk
    with open(pkl_file, 'rb') as f:
        graph = pickle.load(f)
    
    # Get degree for each node
    degrees = dict(graph.degree())
    
    # Create DataFrame with node index and degree
    degree_df = pd.DataFrame({
        'node': list(degrees.keys()),
        'degree': list(degrees.values())
    })
    
    # Save to CSV (use same sanitized name as pkl file)
    csv_filename = pkl_file.stem + '_degrees.csv'
    csv_file = str(degrees_dir / csv_filename)
    degree_df.to_csv(csv_file, index=False)
    print(f"  Saved: {csv_filename}")
    print(f"  Mean degree: {degree_df['degree'].mean():.2f}")
    print(f"  Max degree: {degree_df['degree'].max()}")
    print(f"  Min degree: {degree_df['degree'].min()}")

print(f"\nOutput directory: {output_dir}")
print(f"Degrees directory: {degrees_dir}")
print("\nDone!")
