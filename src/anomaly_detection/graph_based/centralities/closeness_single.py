# Calculate Closeness Centrality for a single Visibility Graph using NetworKit
# Usage: python closeness_single.py <graph_name>
# Example: python closeness_single.py G_NVG_p3_rms

import sys
import pickle
import networkx as nx
import networkit as nk
import pandas as pd
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python closeness_single.py <graph_name>")
    print("Example: python closeness_single.py G_NVG_p3_rms")
    sys.exit(1)

graph_name = sys.argv[1]

# Paths
graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/centralities/closeness')
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

# Calculate approximate closeness centrality with NetworKit
# Using sampling for faster computation on large graphs
n_samples = 2000  # Number of samples for approximation
print(f"\nCalculating approximate closeness centrality (nSamples={n_samples})...")
cc = nk.centrality.ApproxCloseness(nk_graph, nSamples=n_samples, normalized=True)
cc.run()
closeness_scores = cc.scores()

print("Closeness calculation completed!")

# Map back to original node indices
closeness_dict = {nx_nodes[nk_id]: closeness_scores[nk_id] for nk_id in range(len(nx_nodes))}

# Create DataFrame preserving node indices
df_closeness = pd.DataFrame({
    'node': list(closeness_dict.keys()),
    'closeness_centrality': list(closeness_dict.values())
})

# Sort by node index to preserve order
df_closeness = df_closeness.sort_values('node').reset_index(drop=True)

# Save individual CSV
csv_file = output_dir / f"{graph_name}_closeness.csv"
df_closeness.to_csv(csv_file, index=False)

print(f"\nResults:")
print(f"  Nodes: {len(closeness_dict)}")
print(f"  Max closeness: {df_closeness['closeness_centrality'].max():.6f}")
print(f"  Mean closeness: {df_closeness['closeness_centrality'].mean():.6f}")
print(f"  Min closeness: {df_closeness['closeness_centrality'].min():.6f}")
print(f"  Std closeness: {df_closeness['closeness_centrality'].std():.6f}")
print(f"\nSaved: {csv_file}")

print("\n" + "=" * 60)
print("CLOSENESS CALCULATION COMPLETED")
print("=" * 60)
