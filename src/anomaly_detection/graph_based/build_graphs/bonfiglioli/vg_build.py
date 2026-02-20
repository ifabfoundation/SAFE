#!/usr/bin/env python
# coding: utf-8

# Build Natural Visibility Graphs for Bonfiglioli Dataset (not weighted)

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from ts2vg import NaturalVG
import networkx as nx

# Load dataset
data_file = Path('/home/projects/safe/data/bonfiglioli/processed/ETR_946 - Prova 30_5 - 310 - Fatica 336h_1_1.csv')

print(f"Loading dataset from: {data_file}")
df = pd.read_csv(data_file)

print(f"\nDataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {df.columns.tolist()}")

# Use 'time_relative' as timestamp, drop 'datetime' and 'time'
df = df.drop(columns=['datetime', 'time'])
print(f"\nAfter dropping 'datetime' and 'time':")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Get all columns except timestamp
timestamp_col = 'time_relative'

# Target columns to process
target_keys = ['tan_oil', 'temp_oil_marcia', 'temp_pt100_oil', 'vib_mot_marcia_rms', 
               'vib_rid_marc2_rms', 'vib_rid_marcia_rms', 'temp_cassa_riduttore', 'temp_mot_marcia']

# Filter data columns to only include target keys
data_columns = [col for col in df.columns if col != timestamp_col and col in target_keys]

print(f"\n{'='*60}")
print(f"BUILDING VISIBILITY GRAPHS (NVG only)")
print(f"{'='*60}")
print(f"Timestamp column: {timestamp_col}")
print(f"Target columns: {target_keys}")
print(f"Data columns to process ({len(data_columns)}): {data_columns}")

# Output directory
output_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs')
output_dir.mkdir(parents=True, exist_ok=True)

# Storage for all graphs
all_nvg_graphs = {}

# Build visibility graphs for each column
for col in data_columns:
    print(f"\n{'='*60}")
    print(f"Processing column: {col}")
    print(f"{'='*60}")
    
    # Get time series data and skip first 50,000 points
    ts_data_full = df[col].values
    ts_data = ts_data_full[50000:]
    print(f"Original time series length: {len(ts_data_full)}")
    print(f"Time series length (after skipping first 50,000): {len(ts_data)}")
    print(f"Min: {ts_data.min():.4f}, Max: {ts_data.max():.4f}, Mean: {ts_data.mean():.4f}")
    
    # Build Natural Visibility Graph (NVG) only
    print(f"\nBuilding Natural Visibility Graph (NVG)...")
    nvg = NaturalVG()
    nvg.build(ts_data)
    G_nvg = nvg.as_networkx()
    graph_name_nvg = f"G_NVG_{col}"
    all_nvg_graphs[graph_name_nvg] = G_nvg
    
    print(f"  {graph_name_nvg}: {G_nvg.number_of_nodes()} nodes, {G_nvg.number_of_edges()} edges")

# Save all graphs
print(f"\n{'='*60}")
print(f"SAVING GRAPHS (NVG only)")
print(f"{'='*60}")

# Save individual graph files
for graph_name, graph in all_nvg_graphs.items():
    pkl_file = output_dir / f"{graph_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved: {pkl_file.name}")

# Save collection file
collection = {
    'all_nvg_graphs': all_nvg_graphs
}
collection_file = output_dir / 'all_visibility_graphs_collections.pkl'
with open(collection_file, 'wb') as f:
    pickle.dump(collection, f)
print(f"\nSaved collection: {collection_file.name}")

# Print summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total NVG graphs: {len(all_nvg_graphs)}")
print(f"Target columns: {target_keys}")
print(f"Output directory: {output_dir}")
print("\nDone!")

