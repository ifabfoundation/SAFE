# Community Detection using PyIOmica for Visibility Graphs
import pickle
import networkx as nx
import pandas as pd
from pathlib import Path
import numpy as np

print("COMMUNITY DETECTION WITH PYIOMICA")
print("=" * 60)

# Import PyIOmica
try:
    from pyiomica import visibilityGraphCommunityDetection as vgcd
    
except ImportError as e:
    exit(1)

# Paths
input_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/communities')
output_dir.mkdir(parents=True, exist_ok=True)


# Load only the specific graph
print("\nLoading G_NVG_p3_rms graph...")
graphs = {}

selected_graph_file = input_dir / 'G_NVG_p3_rms.pkl'

if selected_graph_file.exists():
    try:
        with open(selected_graph_file, 'rb') as f:
            graph = pickle.load(f)
        
        graph_name = selected_graph_file.stem
        graphs[graph_name] = graph
        
        print(f"Loaded {graph_name}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
    except Exception as e:
        print(f"Error loading {selected_graph_file.name}: {e}")
        exit(1)
else:
    print(f"Error: {selected_graph_file} not found")
    exit(1)

print(f"\nSuccessfully loaded 1 graph")

def detect_communities_pyiomica(graph, graph_name):
    """
    Apply PyIOmica community detection algorithm.
    Parameters:
        graph (networkx.Graph): The visibility graph.
        graph_name (str): Name of the graph for logging.
    Returns:
        pd.DataFrame: DataFrame with node-community assignments.
    """
    
    try:
        # Apply PyIOmica community detection by path length
        communities = vgcd.communityDetectByPathLength(graph)
        
        # Convert to node -> community mapping
        node_communities = {}
        
        if isinstance(communities, dict):
            # Direct node -> community mapping
            node_communities = communities
        elif isinstance(communities, (list, tuple)):
            # List of communities, each containing nodes
            for comm_id, community in enumerate(communities):
                for node in community:
                    node_communities[node] = comm_id
        elif hasattr(communities, 'membership'):
            # Object with membership attribute
            try:
                node_communities = dict(communities.membership)
            except:
                node_communities = dict(communities)
        else:
            # Try to convert to dict
            node_communities = dict(communities)
        
        # Create results dataframe
        results = []
        for node in graph.nodes():
            community_id = node_communities.get(node, -1)  # -1 for unassigned
            
            results.append({
                'node': node,
                'community': community_id
            })
        
        df = pd.DataFrame(results)
        
        num_communities = len(set(node_communities.values()))
        print(f"Communities: {num_communities}")
        
        return df
        
    except Exception as e:
        print(f"Error detecting communities: {e}")
        return None

# Apply community detection to the selected graph

community_results = {}

for graph_name, graph in graphs.items():
    try:
        print(f"\nProcessing {graph_name}...")
        # Detect communities
        result_df = detect_communities_pyiomica(graph, graph_name)
        
        if result_df is not None:
            community_results[graph_name] = result_df
            
            # Save individual results
            csv_file = output_dir / f"{graph_name}_communities.csv"
            result_df.to_csv(csv_file, index=False)
            print(f"Results saved: {csv_file.name}")
        
    except Exception as e:
        print(f"Error processing {graph_name}: {e}")

# Save all results as pickle
results_pickle = output_dir / "community_results.pkl"
with open(results_pickle, 'wb') as f:
    pickle.dump(community_results, f)
print(f"All results saved: {results_pickle.name}")

# Print summary statistics
print(f"\nCOMMUNITY DETECTION COMPLETED")
print(f"Graphs analyzed: {len(community_results)}/{len(graphs)}")
print(f"Output directory: {output_dir}")
print(f"Files saved: {len(community_results)} CSV + 1 pickle")
print("\nCommunity detection completed successfully!")
