#!/usr/bin/env python
# coding: utf-8

# Build Visibility Graphs for p3/p4 rms and temp columns

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

# Select p3/p4 rms and temp columns
# Column format: bonfi/gb1_p3_acc_rms, bonfi/gb1_p3_temp, bonfi/gb1_p4_acc_rms, bonfi/gb1_p4_temp
target_columns = [
    'bonfi/gb1_p3_acc_rms',
    'bonfi/gb1_p3_temp',
    'bonfi/gb1_p4_acc_rms',
    'bonfi/gb1_p4_temp'
]

# Check if columns exist
missing_cols = [col for col in target_columns if col not in df.columns]
if missing_cols:
    print(f"\nWarning: Missing columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

print(f"\nSelected columns: {target_columns}")

# Extract data for selected columns and take only first 10000 rows
df_selected = df[target_columns].head(10000)

print(f"\nSelected data shape: {df_selected.shape}")
print(f"Using first 10000 rows from each time series")

# Build Horizontal Visibility Graphs
print("\n" + "="*60)
print("BUILDING HORIZONTAL VISIBILITY GRAPHS")
print("="*60)

all_hvg_graphs = {}

for col in target_columns:
    print(f"\nProcessing column: {col}")
    
    # Get time series data
    ts_data = df_selected[col].values
    print(f"  Time series length: {len(ts_data)}")
    print(f"  Min: {ts_data.min():.4f}, Max: {ts_data.max():.4f}, Mean: {ts_data.mean():.4f}")
    
    # Build Horizontal Visibility Graph
    print(f"  Building Horizontal Visibility Graph...")
    hvg = HorizontalVG()
    hvg.build(ts_data)
    G_hvg = hvg.as_networkx()
    
    # Create graph name
    # bonfi/gb1_p3_acc_rms -> HVG_p3_rms
    # bonfi/gb1_p3_temp -> HVG_p3_temp
    if 'p3_acc_rms' in col:
        graph_name = 'HVG_p3_rms'
    elif 'p3_temp' in col:
        graph_name = 'HVG_p3_temp'
    elif 'p4_acc_rms' in col:
        graph_name = 'HVG_p4_rms'
    elif 'p4_temp' in col:
        graph_name = 'HVG_p4_temp'
    else:
        graph_name = f"HVG_{col.replace('/', '_')}"
    
    all_hvg_graphs[graph_name] = G_hvg
    
    print(f"  {graph_name}: {G_hvg.number_of_nodes()} nodes, {G_hvg.number_of_edges()} edges")

# Save graphs
output_dir = Path('/home/projects/safe/outputs/networks/grafi/reducted')
output_dir.mkdir(parents=True, exist_ok=True)

print("\n" + "="*60)
print("SAVING GRAPHS")
print("="*60)

for graph_name, graph in all_hvg_graphs.items():
    pkl_file = output_dir / f"{graph_name}.pkl"
    with open(pkl_file, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Saved: {pkl_file.name}")

# Save collection
collection = {'all_hvg_graphs': all_hvg_graphs}
collection_file = output_dir / 'hvg_reducted_collection.pkl'
with open(collection_file, 'wb') as f:
    pickle.dump(collection, f)
print(f"\nSaved collection: {collection_file.name}")

print("\nData ready for visibility graph construction.")
