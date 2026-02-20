#!/usr/bin/env python
# coding: utf-8

# Calculate Approximate Closeness Centrality for Bonfiglioli Graphs using NetworKit

import pickle
import networkx as nx
import networkit as nk
import pandas as pd
from pathlib import Path
import time

# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs')
output_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs/closeness/scores')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Load all graphs
print("\n" + "="*60)
print("LOADING BONFIGLIOLI GRAPHS")
print("="*60)

# Load only NVG graphs
graphs = {}
pickle_files = sorted([f for f in input_dir.glob('NVG_*.pkl')])
print(f"Found {len(pickle_files)} NVG pickle files")

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

# Calculate closeness for each graph
print("\n" + "="*60)
print("CALCULATING APPROXIMATE CLOSENESS CENTRALITY")
print("="*60)

for graph_name, nx_graph in graphs.items():
    print(f"\n{'='*60}")
    print(f"Processing: {graph_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Convert NetworkX to NetworKit
    print("Converting NetworkX -> NetworKit...")
    
    # Create node mapping (NetworkX node -> NetworKit node)
    node_list = list(nx_graph.nodes())
    node_to_nk = {node: i for i, node in enumerate(node_list)}
    
    # Create NetworKit graph
    nk_graph = nk.graph.Graph(n=len(node_list), weighted=False, directed=False)
    
    # Add edges
    for u, v in nx_graph.edges():
        nk_u = node_to_nk[u]
        nk_v = node_to_nk[v]
        nk_graph.addEdge(nk_u, nk_v)
    
    print(f"NetworKit graph: {nk_graph.numberOfNodes()} nodes, {nk_graph.numberOfEdges()} edges")
    
    # Calculate approximate closeness centrality
    print(f"Calculating approximate closeness (nSamples=3000)...")
    
    try:
        # ApproxCloseness(graph, nSamples, normalized=True)
        closeness_algo = nk.centrality.ApproxCloseness(nk_graph, 3000, True)
        closeness_algo.run()
        
        # Get closeness scores for all nodes
        closeness_scores = closeness_algo.scores()
        
        # Create DataFrame with original node indices
        results = []
        for nx_node in node_list:
            nk_node = node_to_nk[nx_node]
            closeness = closeness_scores[nk_node]
            results.append({
                'node': nx_node,
                'closeness': closeness
            })
        
        df_result = pd.DataFrame(results)
        
        # Statistics
        print(f"\nCloseness statistics:")
        print(f"  Mean: {df_result['closeness'].mean():.6f}")
        print(f"  Std: {df_result['closeness'].std():.6f}")
        print(f"  Min: {df_result['closeness'].min():.6f}")
        print(f"  Max: {df_result['closeness'].max():.6f}")
        print(f"  Median: {df_result['closeness'].median():.6f}")
        
        # Top 10 nodes
        top_10 = df_result.nlargest(10, 'closeness')
        print(f"\nTop 10 nodes by closeness:")
        for idx, row in top_10.iterrows():
            print(f"  Node {int(row['node'])}: {row['closeness']:.6f}")
        
        # Save results
        output_file = output_dir / f"{graph_name}_closeness.csv"
        df_result.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Saved: {output_file.name}")
        print(f"  Time elapsed: {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error calculating closeness: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"Results saved to: {output_dir}")
