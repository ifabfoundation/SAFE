#!/usr/bin/env python
# coding: utf-8

# Community Detection for HVG_p3_temp

import pickle
import networkx as nx
from pathlib import Path
import pyiomica.visibilityGraphCommunityDetection as vgcd
import pandas as pd

# Paths
graph_file = Path('/home/projects/safe/outputs/networks/grafi/reducted/HVG_p3_temp.pkl')
output_dir = Path('/home/projects/safe/outputs/networks/communities/reducted')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loading graph from: {graph_file}")

if not graph_file.exists():
    print(f"Error: Graph file not found: {graph_file}")
    exit(1)

# Load graph
with open(graph_file, 'rb') as f:
    G = pickle.load(f)

print(f"\nGraph loaded successfully!")
print(f"Graph: HVG_p3_temp")
print(f"Nodes: {G.number_of_nodes()}")
print(f"Edges: {G.number_of_edges()}")

# Community detection with visibilityGraphCommunityDetection
print("\n" + "="*60)
print("COMMUNITY DETECTION")
print("="*60)

print("\nRunning communityDetectByPathLength...")
communities = vgcd.communityDetectByPathLength(G, outputTimePoints=False)

print(f"\nCommunity detection completed!")
print(f"Type of communities: {type(communities)}")
print(f"Length: {len(communities)}")

# communities is a list of lists, where each sublist is a community
# Convert to node -> community mapping
node_to_community = {}
for comm_id, community_nodes in enumerate(communities):
    for node in community_nodes:
        node_to_community[node] = comm_id

print(f"Number of communities detected: {len(communities)}")
print(f"Total nodes assigned: {len(node_to_community)}")

# Create DataFrame with node -> community mapping
df_communities = pd.DataFrame({
    'node': list(node_to_community.keys()),
    'community': list(node_to_community.values())
})

# Statistics
print("\nCommunity statistics:")
community_sizes = df_communities['community'].value_counts().sort_index()
for comm_id, size in community_sizes.items():
    print(f"  Community {comm_id}: {size} nodes")

# Save results
output_file = output_dir / 'HVG_p3_temp_communities.csv'
df_communities.to_csv(output_file, index=False)
print(f"\n✓ Saved communities to: {output_file}")

print("\nGraph ready for community detection.")
