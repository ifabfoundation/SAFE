# Calculate Average Edge Overlap for Visibility Graphs
# Following Lacasa, Nicosia & Latora formula:
# ω = (Σ_{i,j} Σ_α a_{ij}^{[α]}) / (M × Σ_{i,j} (1 - δ_{0,Σ_α a_{ij}^{[α]}}))
# where a_{ij}^{[α]} = 1 if edge (i,j) exists in layer α, 0 otherwise
# Numerator: total edge occurrences across all layers
# Denominator: M × number of unique edges (appearing in at least one layer)

import pickle
import networkx as nx
from pathlib import Path
from collections import defaultdict

# Paths
graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')
output_dir = Path('/home/projects/safe/outputs/networks/multiplex/average_edge_overlap')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading visibility graphs...")
print(f"Input directory: {graphs_dir}")
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

# Separate NVG and HVG graphs
nvg_graphs = {name: g for name, g in graphs.items() if 'NVG' in name}
hvg_graphs = {name: g for name, g in graphs.items() if 'HVG' in name}

print(f"\nNVG graphs: {len(nvg_graphs)}")
print(f"HVG graphs: {len(hvg_graphs)}")

def calculate_average_edge_overlap_lacasa(graphs_dict):
    """
    Calculate average edge overlap following Lacasa et al. formula.
    ω = (Σ edge occurrences) / (M × unique edges)
    """
    graph_list = list(graphs_dict.values())
    graph_names = list(graphs_dict.keys())
    M = len(graph_list)
    
    if M < 2:
        print("Need at least 2 graphs to calculate average edge overlap")
        return 0.0, {}, 0, 0
    
    print(f"\nComputing average edge overlap for {M} graphs using Lacasa et al. formula...")
    
    # Get all unique edges across all graphs
    all_edges = set()
    for G in graph_list:
        for u, v in G.edges():
            edge = (min(u, v), max(u, v))  # canonical form
            all_edges.add(edge)
    
    print(f"Total unique edges across all graphs: {len(all_edges)}")
    
    # For each edge, count in how many graphs it appears
    edge_counts = defaultdict(int)
    for edge in all_edges:
        for G in graph_list:
            u, v = edge
            if G.has_edge(u, v):
                edge_counts[edge] += 1
    
    # Numerator: Σ_{i,j} Σ_α a_{ij}^{[α]}
    # Total number of edge occurrences across all graphs
    numerator = sum(edge_counts.values())
    
    # Denominator: M × Σ_{i,j} (1 - δ_{0,Σ_α a_{ij}^{[α]}})
    # M × number of edges that appear in at least one graph
    edges_with_nonzero_count = sum(1 for count in edge_counts.values() if count > 0)
    denominator = M * edges_with_nonzero_count
    
    # Average edge overlap
    if denominator > 0:
        average_edge_overlap = numerator / denominator
    else:
        average_edge_overlap = 0.0
    
    print(f"\nLacasa et al. Average Edge Overlap Calculation:")
    print(f"Numerator (total edge occurrences): {numerator}")
    print(f"Denominator (M × unique edges): {denominator}")
    print(f"Average edge overlap: {average_edge_overlap:.6f}")
    
    return average_edge_overlap, dict(edge_counts), numerator, denominator

# Calculate average edge overlap for NVG graphs
print("\n" + "=" * 60)
print("CALCULATING AVERAGE EDGE OVERLAP FOR NVG GRAPHS")
print("=" * 60)

nvg_avg_overlap, nvg_edge_counts, nvg_numerator, nvg_denominator = calculate_average_edge_overlap_lacasa(nvg_graphs)

print(f"\n{'=' * 60}")
print(f"NVG Average Edge Overlap: {nvg_avg_overlap:.6f}")
print(f"{'=' * 60}")

# Calculate average edge overlap for HVG graphs
print("\n" + "=" * 60)
print("CALCULATING AVERAGE EDGE OVERLAP FOR HVG GRAPHS")
print("=" * 60)

hvg_avg_overlap, hvg_edge_counts, hvg_numerator, hvg_denominator = calculate_average_edge_overlap_lacasa(hvg_graphs)

print(f"\n{'=' * 60}")
print(f"HVG Average Edge Overlap: {hvg_avg_overlap:.6f}")
print(f"{'=' * 60}")

# Save results
results = {
    'nvg': {
        'average_edge_overlap': nvg_avg_overlap,
        'edge_counts': nvg_edge_counts,
        'numerator': nvg_numerator,
        'denominator': nvg_denominator,
        'num_graphs': len(nvg_graphs)
    },
    'hvg': {
        'average_edge_overlap': hvg_avg_overlap,
        'edge_counts': hvg_edge_counts,
        'numerator': hvg_numerator,
        'denominator': hvg_denominator,
        'num_graphs': len(hvg_graphs)
    }
}

# Save as pickle
results_file = output_dir / 'average_edge_overlap_results.pkl'
with open(results_file, 'wb') as f:
    pickle.dump(results, f)
print(f"\n Results saved: {results_file}")

# Save summary as text
summary_file = output_dir / 'average_edge_overlap_summary.txt'
with open(summary_file, 'w') as f:
    f.write("AVERAGE EDGE OVERLAP RESULTS (Lacasa et al. formula)\n")
    f.write("=" * 60 + "\n\n")
    
    f.write(f"NVG Graphs ({len(nvg_graphs)} graphs):\n")
    f.write(f"  Average Edge Overlap: {nvg_avg_overlap:.6f}\n")
    f.write(f"  Numerator: {nvg_numerator}\n")
    f.write(f"  Denominator: {nvg_denominator}\n\n")
    
    f.write(f"HVG Graphs ({len(hvg_graphs)} graphs):\n")
    f.write(f"  Average Edge Overlap: {hvg_avg_overlap:.6f}\n")
    f.write(f"  Numerator: {hvg_numerator}\n")
    f.write(f"  Denominator: {hvg_denominator}\n\n")

print(f" Summary saved: {summary_file}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"NVG Average Edge Overlap: {nvg_avg_overlap:.6f}")
print(f"HVG Average Edge Overlap: {hvg_avg_overlap:.6f}")
print(f"\nOutput directory: {output_dir}")
print("Done!")
