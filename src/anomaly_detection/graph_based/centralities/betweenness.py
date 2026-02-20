
# Calculate Betweenness Centrality for Visibility Graphs using NetworKit (Approximate)

import pickle
import networkx as nx
import networkit as nk
import pandas as pd
from pathlib import Path

# Paths
graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/centralities/betweenness')
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

# Calculate betweenness centrality for each graph
print("\n" + "=" * 60)
print("CALCULATING APPROXIMATE BETWEENNESS CENTRALITY (1500 samples)")
print("=" * 60)

betweenness_results = {}

for graph_name, graph in graphs.items():
    print(f"\nProcessing {graph_name}...")
    
    try:
        # Convert NetworkX graph to NetworKit graph
        # Create node mapping (NetworkX node -> NetworKit node)
        nx_nodes = list(graph.nodes())
        node_mapping = {nx_node: nk_id for nk_id, nx_node in enumerate(nx_nodes)}
        
        # Create NetworKit graph
        nk_graph = nk.Graph(len(nx_nodes), weighted=False, directed=False)
        
        # Add edges
        for u, v in graph.edges():
            nk_graph.addEdge(node_mapping[u], node_mapping[v])
        
        print(f"  Converted to NetworKit: {nk_graph.numberOfNodes()} nodes, {nk_graph.numberOfEdges()} edges")
        
        # Calculate approximate betweenness centrality with NetworKit
        bc = nk.centrality.ApproxBetweenness(nk_graph, nSamples=1500)
        bc.run()
        betweenness_scores = bc.scores()
        
        # Map back to original node indices
        betweenness_dict = {nx_nodes[nk_id]: betweenness_scores[nk_id] for nk_id in range(len(nx_nodes))}
        
        # Create DataFrame preserving node indices
        df_betweenness = pd.DataFrame({
            'node': list(betweenness_dict.keys()),
            'betweenness_centrality': list(betweenness_dict.values())
        })
        
        # Sort by node index (not by betweenness) to preserve order
        df_betweenness = df_betweenness.sort_values('node').reset_index(drop=True)
        
        # Store results
        betweenness_results[graph_name] = df_betweenness
        
        # Save individual CSV
        csv_file = output_dir / f"{graph_name}_betweenness.csv"
        df_betweenness.to_csv(csv_file, index=False)
        
        print(f"  Nodes: {len(betweenness_dict)}")
        print(f"  Max betweenness: {df_betweenness['betweenness_centrality'].max():.6f}")
        print(f"  Mean betweenness: {df_betweenness['betweenness_centrality'].mean():.6f}")
        print(f"  Saved: {csv_file.name}")
        
    except Exception as e:
        print(f"  Error calculating betweenness for {graph_name}: {e}")
        continue

# Save all results as pickle
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

results_pickle = output_dir / "betweenness_results.pkl"
with open(results_pickle, 'wb') as f:
    pickle.dump(betweenness_results, f)
print(f"All results saved: {results_pickle}")

# Create summary statistics
print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

summary_data = []
for graph_name, df in betweenness_results.items():
    summary_data.append({
        'graph': graph_name,
        'num_nodes': len(df),
        'max_betweenness': df['betweenness_centrality'].max(),
        'mean_betweenness': df['betweenness_centrality'].mean(),
        'median_betweenness': df['betweenness_centrality'].median(),
        'std_betweenness': df['betweenness_centrality'].std()
    })

summary_df = pd.DataFrame(summary_data)
summary_file = output_dir / "betweenness_summary.csv"
summary_df.to_csv(summary_file, index=False)

print(f"\nSummary statistics:")
print(summary_df.to_string(index=False))
print(f"\nSummary saved: {summary_file}")

print("\n" + "=" * 60)
print("BETWEENNESS CENTRALITY CALCULATION COMPLETED")
print("=" * 60)
print(f"Graphs processed: {len(betweenness_results)}")
print(f"Output directory: {output_dir}")
print("The end")
