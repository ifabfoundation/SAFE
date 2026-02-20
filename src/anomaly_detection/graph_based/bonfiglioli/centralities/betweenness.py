#!/usr/bin/env python
# coding: utf-8

# Calculate Approximate Betweenness Centrality for Bonfiglioli Graphs using NetworKit

import pickle
import networkx as nx
import networkit as nk
import pandas as pd
from pathlib import Path
import time

# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs')
output_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs/betweenness/scores')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Input directory: {input_dir}")
print(f"Output directory: {output_dir}")

# Load all NVG graphs
print("\n" + "="*60)
print("LOADING BONFIGLIOLI GRAPHS")
print("="*60)

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

# Calculate betweenness for each graph
print("\n" + "="*60)
print("CALCULATING APPROXIMATE BETWEENNESS CENTRALITY")
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
    
    # Calculate approximate betweenness centrality
    print(f"Calculating approximate betweenness (nSamples=1500)...")
    
    try:
        # ApproxBetweenness(graph, nSamples) - positional parameter
        betweenness_algo = nk.centrality.ApproxBetweenness(nk_graph, 1500)
        betweenness_algo.run()
        
        # Get betweenness scores for all nodes
        betweenness_scores = betweenness_algo.scores()
        
        # Create DataFrame with original node indices
        results = []
        for nx_node in node_list:
            nk_node = node_to_nk[nx_node]
            betweenness = betweenness_scores[nk_node]
            results.append({
                'node': nx_node,
                'betweenness': betweenness
            })
        
        df_result = pd.DataFrame(results)
        
        # Statistics
        print(f"\nBetweenness statistics:")
        print(f"  Mean: {df_result['betweenness'].mean():.6f}")
        print(f"  Std: {df_result['betweenness'].std():.6f}")
        print(f"  Min: {df_result['betweenness'].min():.6f}")
        print(f"  Max: {df_result['betweenness'].max():.6f}")
        print(f"  Median: {df_result['betweenness'].median():.6f}")
        
        # Top 10 nodes
        top_10 = df_result.nlargest(10, 'betweenness')
        print(f"\nTop 10 nodes by betweenness:")
        for idx, row in top_10.iterrows():
            print(f"  Node {int(row['node'])}: {row['betweenness']:.6f}")
        
        # Save results
        output_file = output_dir / f"{graph_name}_betweenness.csv"
        df_result.to_csv(output_file, index=False)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Saved: {output_file.name}")
        print(f"  Time elapsed: {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"\n✗ Error calculating betweenness: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*60)
print("DONE")
print("="*60)
print(f"Results saved to: {output_dir}")
