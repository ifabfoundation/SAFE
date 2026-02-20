# Calculate Betweenness Centrality for a single Visibility Graph using NetworKit (Approximate)
# Usage: python betweenness_single.py <graph_name>
# Example: python betweenness_single.py G_NVG_p3_rms

import sys
import pickle
import networkx as nx
import networkit as nk
import pandas as pd
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python betweenness_single.py <graph_name>")
    print("Example: python betweenness_single.py G_NVG_p3_rms")
    sys.exit(1)

graph_name = sys.argv[1]

# Paths
graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/centralities/betweenness')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Processing graph: {graph_name}")
print(f"Input directory: {graphs_dir}")
print(f"Output directory: {output_dir}")
print("=" * 60)

# Load the specific graph
pkl_file = graphs_dir / f"{graph_name}.pkl"

if not pkl_file.exists():
    print(f"ERROR: Graph file not found: {pkl_file}")
    sys.exit(1)

print(f"Loading {pkl_file.name}...")
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

# Handle both single graph and dictionary of graphs
if isinstance(data, dict):
    if graph_name in data:
        graph = data[graph_name]
    else:
        # Try to find the graph in the dict
        if len(data) == 1:
            graph = list(data.values())[0]
        else:
            print(f"ERROR: Graph {graph_name} not found in pickle file")
            sys.exit(1)
elif hasattr(data, 'number_of_nodes'):
    graph = data
else:
    print(f"ERROR: Invalid data format in {pkl_file}")
    sys.exit(1)

print(f"Loaded {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

# Convert NetworkX graph to NetworKit graph
print("\nConverting to NetworKit...")
nx_nodes = list(graph.nodes())
node_mapping = {nx_node: nk_id for nk_id, nx_node in enumerate(nx_nodes)}

nk_graph = nk.Graph(len(nx_nodes), weighted=False, directed=False)

for u, v in graph.edges():
    nk_graph.addEdge(node_mapping[u], node_mapping[v])

print(f"Converted: {nk_graph.numberOfNodes()} nodes, {nk_graph.numberOfEdges()} edges")

# Calculate approximate betweenness centrality with NetworKit
print(f"\nCalculating approximate betweenness centrality (nSamples=1500)...")
bc = nk.centrality.ApproxBetweenness(nk_graph, 1500)
bc.run()
betweenness_scores = bc.scores()

print("Betweenness calculation completed!")

# Map back to original node indices
betweenness_dict = {nx_nodes[nk_id]: betweenness_scores[nk_id] for nk_id in range(len(nx_nodes))}

# Create DataFrame preserving node indices
df_betweenness = pd.DataFrame({
    'node': list(betweenness_dict.keys()),
    'betweenness_centrality': list(betweenness_dict.values())
})

# Sort by node index to preserve order
df_betweenness = df_betweenness.sort_values('node').reset_index(drop=True)

# Save individual CSV
csv_file = output_dir / f"{graph_name}_betweenness.csv"
df_betweenness.to_csv(csv_file, index=False)

print(f"\nResults:")
print(f"  Nodes: {len(betweenness_dict)}")
print(f"  Max betweenness: {df_betweenness['betweenness_centrality'].max():.6f}")
print(f"  Mean betweenness: {df_betweenness['betweenness_centrality'].mean():.6f}")
print(f"  Min betweenness: {df_betweenness['betweenness_centrality'].min():.6f}")
print(f"  Std betweenness: {df_betweenness['betweenness_centrality'].std():.6f}")
print(f"\nSaved: {csv_file}")

print("\n" + "=" * 60)
print("BETWEENNESS CALCULATION COMPLETED")
print("=" * 60)
