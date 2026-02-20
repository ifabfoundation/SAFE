#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


# In[ ]:


df = pd.read_csv('/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv', index_col=0)


# In[ ]:


df1 = df[ 'bonfi/gb1_p3_acc_rms']
df2 = df['bonfi/gb1_p3_acc_cfa']
df3 = df['bonfi/gb1_p3_acc_kurt']
df4 = df['bonfi/gb1_p3_acc_max']
df5 = df['bonfi/gb1_p3_acc_min']
df6 = df['bonfi/gb1_p3_acc_skew']
df7 = df['bonfi/gb1_p3_acc_std']
df8 = df['bonfi/gb1_p3_temp']

df1_4 = df['bonfi/gb1_p4_acc_rms']
df2_4 = df['bonfi/gb1_p4_acc_cfa']
df3_4 = df['bonfi/gb1_p4_acc_kurt']
df4_4 = df['bonfi/gb1_p4_acc_max']
df5_4 = df['bonfi/gb1_p4_acc_min']
df6_4 = df['bonfi/gb1_p4_acc_skew']
df7_4 = df['bonfi/gb1_p4_acc_std']
df8_4 = df['bonfi/gb1_p4_temp']


# In[ ]:


# Build NVG and HVG graphs for all 16 time series (8 from p3 sensor + 8 from p4 sensor)
# Total: 32 graphs (16 NVG + 16 HVG)

try:
    from ts2vg import NaturalVG, HorizontalVG
    import networkx as nx
    import numpy as np

    # Define all time series variables and their corresponding names
    p3_series = [
        ('df1', 'p3_rms'),
        ('df2', 'p3_cfa'), 
        ('df3', 'p3_kurt'),
        ('df4', 'p3_max'),
        ('df5', 'p3_min'),
        ('df6', 'p3_skew'),
        ('df7', 'p3_std'),
        ('df8', 'p3_temp')
    ]

    p4_series = [
        ('df1_4', 'p4_rms'),
        ('df2_4', 'p4_cfa'),
        ('df3_4', 'p4_kurt'),
        ('df4_4', 'p4_max'),
        ('df5_4', 'p4_min'),
        ('df6_4', 'p4_skew'),
        ('df7_4', 'p4_std'),
        ('df8_4', 'p4_temp')
    ]

    all_series = p3_series + p4_series

    # Verify all variables exist
    for var_name, _ in all_series:
        if var_name not in globals():
            raise NameError(f"Variable {var_name} not found. Make sure all time series are defined.")

    print(f'Building NVG and HVG graphs for {len(all_series)} time series...')

    # Storage for all graphs
    all_nvg_graphs = {} #dizionario di tutti i grafi NVG
    all_hvg_graphs = {} #dizionario di tutti i grafi HVG
    nvg_list = [] 
    hvg_list = []
    graph_names = []

    # Build graphs for all time series
    for var_name, series_name in all_series:
        # Get the time series data
        ts_data = globals()[var_name]
        ts_array = ts_data.to_numpy()

        # Create graph names
        nvg_name = f'G_NVG_{series_name}'
        hvg_name = f'G_HVG_{series_name}'

        try:
            # Build Natural Visibility Graph
            vg_natural = NaturalVG()
            vg_natural.build(ts_array)
            G_nvg = vg_natural.as_networkx()

            # Add ts_index attribute for consistency
            for pos, node in enumerate(list(G_nvg.nodes())):
                G_nvg.nodes[node]['ts_index'] = pos
                G_nvg.nodes[node]['series_name'] = series_name

            # Store in globals and collections
            globals()[nvg_name] = G_nvg
            all_nvg_graphs[nvg_name] = G_nvg
            nvg_list.append(G_nvg)

            print(f'  {nvg_name}: {G_nvg.number_of_nodes()} nodes, {G_nvg.number_of_edges()} edges')

            # Build Horizontal Visibility Graph
            vg_horizontal = HorizontalVG()
            vg_horizontal.build(ts_array)
            G_hvg = vg_horizontal.as_networkx()

            # Add ts_index attribute for consistency
            for pos, node in enumerate(list(G_hvg.nodes())):
                G_hvg.nodes[node]['ts_index'] = pos
                G_hvg.nodes[node]['series_name'] = series_name

            # Store in globals and collections
            globals()[hvg_name] = G_hvg
            all_hvg_graphs[hvg_name] = G_hvg
            hvg_list.append(G_hvg)

            print(f'  {hvg_name}: {G_hvg.number_of_nodes()} nodes, {G_hvg.number_of_edges()} edges')

            graph_names.append(series_name)

        except Exception as e:
            print(f'Error building graphs for {series_name}: {e}')
            continue

    # Create comprehensive lists and dictionaries
    globals()['all_nvg_graphs'] = all_nvg_graphs
    globals()['all_hvg_graphs'] = all_hvg_graphs
    globals()['nvg_list_all'] = nvg_list
    globals()['hvg_list_all'] = hvg_list
    globals()['graph_names_all'] = graph_names

    # Separate P3 and P4 lists for convenience
    p3_nvg_list = nvg_list[:8]  # First 8 are P3
    p4_nvg_list = nvg_list[8:]  # Last 8 are P4
    p3_hvg_list = hvg_list[:8]
    p4_hvg_list = hvg_list[8:]

    globals()['p3_nvg_list'] = p3_nvg_list
    globals()['p4_nvg_list'] = p4_nvg_list
    globals()['p3_hvg_list'] = p3_hvg_list
    globals()['p4_hvg_list'] = p4_hvg_list

    # Summary statistics
    print(f'\n=== SUMMARY ===')
    print(f'Total graphs created: {len(nvg_list) + len(hvg_list)} (16 NVG + 16 HVG)')
    print(f'P3 sensor: {len(p3_nvg_list) + len(p3_hvg_list)} graphs | P4 sensor: {len(p4_nvg_list) + len(p4_hvg_list)} graphs')

    print(f'\nMain collections created:')
    print(f'- p3_nvg_list, p4_nvg_list: NVG graphs by sensor')
    print(f'- p3_hvg_list, p4_hvg_list: HVG graphs by sensor')
    print(f'- nvg_list_all, hvg_list_all: all graphs combined')

    print(f'\n✅ All 32 graphs (16 NVG + 16 HVG) successfully created and stored!')

except Exception as e:
    print(f'❌ Error during graph construction: {e}')
    import traceback
    traceback.print_exc()


# ## Salvataggio grafi su disco
# 
# Salvataggio di tutti i 32 grafi (16 NVG + 16 HVG) come file pickle nella cartella outputs.

# In[ ]:


# Save all 32 graphs (16 NVG + 16 HVG) as pickle files
import pickle
import os
from datetime import datetime

try:
    # Create output directory
    output_dir = '/home/projects/safe/outputs/networks/grafi'
    os.makedirs(output_dir, exist_ok=True)

    print(f'Saving all graphs to: {output_dir}')
    print('=' * 50)

    saved_files = []

    # Save individual NVG graphs
    print('Saving NVG graphs...')
    for name, graph in all_nvg_graphs.items():
        filename = f'{name}.pkl'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)

        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        print(f'  ✅ {filename}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges ({file_size:.1f} MB)')
        saved_files.append(filename)

    # Save individual HVG graphs
    print('\nSaving HVG graphs...')
    for name, graph in all_hvg_graphs.items():
        filename = f'{name}.pkl'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)

        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Size in MB
        print(f'  ✅ {filename}: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges ({file_size:.1f} MB)')
        saved_files.append(filename)

    # Save all collections in a single file
    print('\nSaving graph collections...')
    collections = {
        'all_nvg_graphs': all_nvg_graphs,
        'all_hvg_graphs': all_hvg_graphs,
        'nvg_list_all': nvg_list,
        'hvg_list_all': hvg_list,
        'p3_nvg_list': p3_nvg_list,
        'p4_nvg_list': p4_nvg_list,
        'p3_hvg_list': p3_hvg_list,
        'p4_hvg_list': p4_hvg_list,
        'graph_names_all': graph_names,
        'metadata': {
            'creation_date': datetime.now().isoformat(),
            'total_graphs': len(nvg_list) + len(hvg_list),
            'nvg_count': len(nvg_list),
            'hvg_count': len(hvg_list),
            'p3_series': len(p3_nvg_list),
            'p4_series': len(p4_nvg_list)
        }
    }

    collections_file = os.path.join(output_dir, 'all_visibility_graphs_collections.pkl')
    with open(collections_file, 'wb') as f:
        pickle.dump(collections, f)

    collections_size = os.path.getsize(collections_file) / (1024 * 1024)
    print(f'  ✅ all_visibility_graphs_collections.pkl ({collections_size:.1f} MB)')
    saved_files.append('all_visibility_graphs_collections.pkl')

    # Calculate total disk usage
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in saved_files) / (1024 * 1024)

    # Summary
    print('\n' + '=' * 50)
    print('📁 SAVE SUMMARY')
    print(f'📍 Location: {output_dir}')
    print(f'📊 Files saved: {len(saved_files)}')
    print(f'💾 Total size: {total_size:.1f} MB')
    print(f'🔢 Graphs saved: {len(all_nvg_graphs)} NVG + {len(all_hvg_graphs)} HVG = {len(all_nvg_graphs) + len(all_hvg_graphs)} total')

    print('\n✅ All visibility graphs successfully saved to disk!')

    # Create a summary file with file listing
    summary_file = os.path.join(output_dir, 'graphs_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Visibility Graphs Save Summary\n")
        f.write(f"Generated on: {datetime.now().isoformat()}\n")
        f.write(f"Location: {output_dir}\n\n")
        f.write(f"Total files: {len(saved_files)}\n")
        f.write(f"Total size: {total_size:.1f} MB\n\n")
        f.write("Files saved:\n")
        for filename in sorted(saved_files):
            f.write(f"  - {filename}\n")

    print(f'📝 Summary saved to: graphs_summary.txt')

except Exception as e:
    print(f'❌ Error saving graphs: {e}')
    import traceback
    traceback.print_exc()

