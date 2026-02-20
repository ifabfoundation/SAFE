#!/usr/bin/env python
# coding: utf-8

# # Visibility Graphs Analysis build from time series
#
# - upload of 32 graphs saved as pickle files (16 NVG + 16 HVG)
# - 8 time series from sensor P3: rms, cfa, kurt, max, min, skew, std, temp
# - 8 time series from sensor P4: rms, cfa, kurt, max, min, skew, std, temp
# 
# **Objectives:** 
# - Loading saved graphs
# - Degree distribution computations for each graph
# - Average degree calculations
# - Saving results in pickle and JSON files


# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pickle
import os
from pathlib import Path
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


# ## 1. Graphs upload


# Load all visibility graphs from saved pickle files
import pickle
import os
from pathlib import Path

def load_visibility_graphs():
    """
    Load all visibility graphs from pickle files.
    Returns both individual graphs and organized collections.
    """

    # Path to saved graphs
    graphs_dir = Path('/home/projects/safe/outputs/networks/grafi')

    if not graphs_dir.exists():
        raise FileNotFoundError(f"Directory {graphs_dir} not found. Run visibility_graphs_build first.")

    print(f"Loading graphs from: {graphs_dir}")
    print("=" * 60)

    # Method 1: Load from collections file (fastest for getting everything)
    collections_file = graphs_dir / 'all_visibility_graphs_collections.pkl'

    if collections_file.exists():
        print("Loading from collections file...")
        with open(collections_file, 'rb') as f:
            data = pickle.load(f)

        # Extract all collections
        all_nvg_graphs = data['all_nvg_graphs']
        all_hvg_graphs = data['all_hvg_graphs']
        nvg_list_all = data['nvg_list_all']
        hvg_list_all = data['hvg_list_all']
        p3_nvg_list = data['p3_nvg_list']
        p4_nvg_list = data['p4_nvg_list']
        p3_hvg_list = data['p3_hvg_list']
        p4_hvg_list = data['p4_hvg_list']
        graph_names_all = data['graph_names_all']
        metadata = data['metadata']

        print(f"Loaded {len(all_nvg_graphs)} NVG graphs")
        print(f"Loaded {len(all_hvg_graphs)} HVG graphs")
        print(f"Total graphs: {metadata['total_graphs']}")
        print(f"Created: {metadata['creation_date']}")

        return {
            'all_nvg_graphs': all_nvg_graphs,
            'all_hvg_graphs': all_hvg_graphs,
            'nvg_list': nvg_list_all,
            'hvg_list': hvg_list_all,
            'p3_nvg_list': p3_nvg_list,
            'p4_nvg_list': p4_nvg_list,
            'p3_hvg_list': p3_hvg_list,
            'p4_hvg_list': p4_hvg_list,
            'graph_names': graph_names_all,
            'metadata': metadata
        }

    else:
        # Method 2: Load individual files (if collections file doesn't exist)
        print("Loading individual graph files...")

        all_nvg_graphs = {}
        all_hvg_graphs = {}

        # Load all pickle files
        pkl_files = list(graphs_dir.glob("G_*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No graph files found in {graphs_dir}")

        for pkl_file in pkl_files:
            print(f"Loading {pkl_file.name}...")
            with open(pkl_file, 'rb') as f:
                graph = pickle.load(f)

            if 'NVG' in pkl_file.name:
                all_nvg_graphs[pkl_file.stem] = graph
            elif 'HVG' in pkl_file.name:
                all_hvg_graphs[pkl_file.stem] = graph

        print(f"Loaded {len(all_nvg_graphs)} NVG graphs")
        print(f"Loaded {len(all_hvg_graphs)} HVG graphs")

        # Organize into lists (recreate the organization)
        nvg_list = list(all_nvg_graphs.values())
        hvg_list = list(all_hvg_graphs.values())

        # Separate by sensor (assuming naming convention G_NVG_p3_* and G_NVG_p4_*)
        p3_nvg_list = [g for name, g in all_nvg_graphs.items() if '_p3_' in name]
        p4_nvg_list = [g for name, g in all_nvg_graphs.items() if '_p4_' in name]
        p3_hvg_list = [g for name, g in all_hvg_graphs.items() if '_p3_' in name]
        p4_hvg_list = [g for name, g in all_hvg_graphs.items() if '_p4_' in name]

        return {
            'all_nvg_graphs': all_nvg_graphs,
            'all_hvg_graphs': all_hvg_graphs,
            'nvg_list': nvg_list,
            'hvg_list': hvg_list,
            'p3_nvg_list': p3_nvg_list,
            'p4_nvg_list': p4_nvg_list,
            'p3_hvg_list': p3_hvg_list,
            'p4_hvg_list': p4_hvg_list,
            'metadata': None
        }

# Load the graphs
try:
    loaded_data = load_visibility_graphs()

    # Extract to global variables for easy access
    all_nvg_graphs = loaded_data['all_nvg_graphs']
    all_hvg_graphs = loaded_data['all_hvg_graphs']
    nvg_list = loaded_data['nvg_list']
    hvg_list = loaded_data['hvg_list']
    p3_nvg_list = loaded_data['p3_nvg_list']
    p4_nvg_list = loaded_data['p4_nvg_list']
    p3_hvg_list = loaded_data['p3_hvg_list']
    p4_hvg_list = loaded_data['p4_hvg_list']
    metadata = loaded_data.get('metadata', None)

    print("\n GRAPHS SUCCESSFULLY LOADED!")
    print(f"Available collections:")
    print(f"  - all_nvg_graphs: dict with {len(all_nvg_graphs)} NVG graphs")
    print(f"  - all_hvg_graphs: dict with {len(all_hvg_graphs)} HVG graphs")
    print(f"  - p3_nvg_list: {len(p3_nvg_list)} P3 NVG graphs")
    print(f"  - p4_nvg_list: {len(p4_nvg_list)} P4 NVG graphs")
    print(f"  - p3_hvg_list: {len(p3_hvg_list)} P3 HVG graphs")
    print(f"  - p4_hvg_list: {len(p4_hvg_list)} P4 HVG graphs")

except FileNotFoundError as e:
    print(f" {e}")
    print("You need to run the graph construction first.")
except Exception as e:
    print(f" Error loading graphs: {e}")
    import traceback
    traceback.print_exc()


# ## 2. Degree Distribution Analysis



from collections import Counter
import numpy as np


print("DEGREE DISTRIBUTIONS FOR ALL GRAPHS")
print("=" * 60)

degree_distributions = {}

# Process NVG graphs
print("\nNATURAL VISIBILITY GRAPHS (NVG):")
for name, graph in all_nvg_graphs.items():
    degrees = [graph.degree(node) for node in graph.nodes()]
    degree_counts = Counter(degrees)

    degree_distributions[name] = {
        'degrees': degrees,
        'degree_counts': dict(degree_counts),
        'graph_type': 'NVG'
    }

    print(f"{name}: degree range [{min(degrees)}-{max(degrees)}], distribution: {dict(list(degree_counts.items())[:5])}...")

# Process HVG graphs
print("\nHORIZONTAL VISIBILITY GRAPHS (HVG):")
for name, graph in all_hvg_graphs.items():
    degrees = [graph.degree(node) for node in graph.nodes()]
    degree_counts = Counter(degrees)

    degree_distributions[name] = {
        'degrees': degrees,
        'degree_counts': dict(degree_counts),
        'graph_type': 'HVG'
    }

    print(f"{name}: degree range [{min(degrees)}-{max(degrees)}], distribution: {dict(list(degree_counts.items())[:5])}...")

print(f"\n Degree distributions calculated for {len(degree_distributions)} graphs")


# ## 3. Average Degree Analysis


# Calculate average degree for each graph
import numpy as np

print("AVERAGE DEGREE FOR ALL GRAPHS")
print("=" * 50)

average_degrees = {}

# Calculate average degree for each graph
for name, data in degree_distributions.items():
    degrees = data['degrees']
    avg_degree = np.mean(degrees)
    average_degrees[name] = avg_degree

    graph_type = data['graph_type']
    print(f"{name}: {avg_degree:.4f} ({graph_type})")

print(f"\n Average degrees calculated for {len(average_degrees)} graphs")


# ## 4. Results Saving


# Save degree distributions and average degrees to files
import pickle
import json
import os
from datetime import datetime

# Create output directory
output_dir = '/home/projects/safe/outputs/networks/degree'
os.makedirs(output_dir, exist_ok=True)

print(f"Saving degree analysis results to: {output_dir}")
print("=" * 60)

try:
    # 1. Save degree_distributions as pickle
    with open(os.path.join(output_dir, 'degree_distributions.pkl'), 'wb') as f:
        pickle.dump(degree_distributions, f)
    print(" degree_distributions.pkl saved")

    # 2. Save average_degrees as pickle
    with open(os.path.join(output_dir, 'average_degrees.pkl'), 'wb') as f:
        pickle.dump(average_degrees, f)
    print("average_degrees.pkl saved")

    # 3. Save degree_distributions as JSON (for readability)
    # Convert numpy arrays to lists for JSON serialization
    degree_distributions_json = {}
    for name, data in degree_distributions.items():
        degree_distributions_json[name] = {
            'degree_counts': data['degree_counts'],
            'graph_type': data['graph_type'],
            'min_degree': min(data['degrees']),
            'max_degree': max(data['degrees']),
            'total_nodes': len(data['degrees'])
        }

    with open(os.path.join(output_dir, 'degree_distributions.json'), 'w') as f:
        json.dump(degree_distributions_json, f, indent=2)
    print(" degree_distributions.json saved")

    # 4. Save average_degrees as JSON
    with open(os.path.join(output_dir, 'average_degrees.json'), 'w') as f:
        json.dump(average_degrees, f, indent=2)
    print("average_degrees.json saved")

    # 5. Create comprehensive summary file
    summary = {
        'analysis_metadata': {
            'creation_date': datetime.now().isoformat(),
            'total_graphs': len(degree_distributions),
            'nvg_graphs': sum(1 for data in degree_distributions.values() if data['graph_type'] == 'NVG'),
            'hvg_graphs': sum(1 for data in degree_distributions.values() if data['graph_type'] == 'HVG'),
        },
        'degree_statistics': {
            'overall': {
                'min_avg_degree': min(average_degrees.values()),
                'max_avg_degree': max(average_degrees.values()),
                'mean_avg_degree': sum(average_degrees.values()) / len(average_degrees)
            },
            'by_graph_type': {
                'nvg_avg_degrees': [avg for name, avg in average_degrees.items() if 'NVG' in name],
                'hvg_avg_degrees': [avg for name, avg in average_degrees.items() if 'HVG' in name]
            }
        },
        'graph_details': {}
    }

    # Add individual graph details
    for name, data in degree_distributions.items():
        avg_deg = average_degrees[name]
        summary['graph_details'][name] = {
            'graph_type': data['graph_type'],
            'average_degree': avg_deg,
            'min_degree': min(data['degrees']),
            'max_degree': max(data['degrees']),
            'total_nodes': len(data['degrees']),
            'unique_degrees': len(data['degree_counts'])
        }

    with open(os.path.join(output_dir, 'degree_analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(" degree_analysis_summary.json saved")

    # Calculate and display file sizes
    files_saved = [
        'degree_distributions.pkl',
        'average_degrees.pkl', 
        'degree_distributions.json',
        'average_degrees.json',
        'degree_analysis_summary.json'
    ]

    total_size = 0
    print(f"\n FILES SAVED:")
    for filename in files_saved:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            total_size += size_mb
            print(f"  📄 {filename}: {size_mb:.2f} MB")

    print(f"\n Total size: {total_size:.2f} MB")
    print(f" Location: {output_dir}")
    print(f" All degree analysis results saved successfully!")

except Exception as e:
    print(f" Error saving results: {e}")
    import traceback
    traceback.print_exc()

