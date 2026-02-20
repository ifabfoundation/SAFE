# Plot Top 20 Oddball Nodes on Original Time Series for Bonfiglioli Graphs

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores/cliquestar_scores')
data_file = Path('/home/projects/safe/data/bonfiglioli/processed/ETR_946 - Prova 30_5 - 310 - Fatica 336h_1_1.csv')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_vs_timeseries/cliquestar')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading Bonfiglioli dataset...")
df_data = pd.read_csv(data_file)
df_data = df_data.drop(columns=['datetime', 'time'])
print(f"Dataset shape: {df_data.shape}")
print(f"Columns: {df_data.columns.tolist()}")

print("\nFinding oddball CSV files...")
oddball_files = list(oddball_dir.glob('*_oddball_results.csv'))
print(f"Found {len(oddball_files)} oddball files")

# Map graph names to column names
# Graph format: NVG_colname or HVG_colname
def graph_to_column(graph_name):
    """Extract column name from graph name (remove NVG_ or HVG_ prefix)"""
    if graph_name.startswith('NVG_'):
        return graph_name[4:]  # Remove 'NVG_'
    elif graph_name.startswith('HVG_'):
        return graph_name[4:]  # Remove 'HVG_'
    return graph_name

print("\n" + "=" * 60)
print("PLOTTING TOP 100 ODDBALL NODES ON TIME SERIES")
print("=" * 60)

# Process each graph
for oddball_file in oddball_files:
    graph_name = oddball_file.stem.replace('_oddball_results', '')
    print(f"\nProcessing {graph_name}...")
    
    # Get corresponding column name
    col_name = graph_to_column(graph_name)
    
    if col_name not in df_data.columns:
        print(f"  Warning: Column '{col_name}' not found in dataset")
        print(f"  Available columns: {df_data.columns.tolist()}")
        continue
    
    # Load oddball results
    df_oddball = pd.read_csv(oddball_file)
    
    # Get top 100 nodes by oddball score
    top_100 = df_oddball.nlargest(100, 'oddball_score')
    print(f"  Top 100 oddball nodes: {top_100['node'].tolist()[:10]}... (showing first 10)")
    
    # Get time series data (skip first 50,000 points to match graph construction)
    ts_data_full = df_data[col_name].values
    ts_data = ts_data_full[50000:]
    time_indices = range(len(ts_data))
    print(f"  Time series: original length {len(ts_data_full)}, using {len(ts_data)} points (skipped first 50,000)")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot time series in gray
    ax.plot(time_indices, ts_data, color='dimgray', alpha=0.7, linewidth=0.8, label='Time series')
    
    # Highlight top 100 oddball nodes with colormap
    top_nodes = top_100['node'].values.astype(int)
    top_scores = top_100['oddball_score'].values
    
    scatter = ax.scatter(top_nodes, ts_data[top_nodes], 
                        c=top_scores, cmap='hot_r', s=80, alpha=0.9, 
                        edgecolors='black', linewidth=0.5,
                        label=f'Top 100 Oddball nodes', zorder=5)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Oddball Score', rotation=270, labelpad=20, fontsize=11)
    
    ax.set_xlabel('Node Index (Time)', fontsize=12)
    ax.set_ylabel(f'{col_name}', fontsize=12)
    ax.set_title(f'Top 100 Oddball Nodes - {graph_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plot_file = output_dir / f"{graph_name}_oddball_vs_timeseries.png"
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {plot_file.name}")

print("\n" + "=" * 60)
print("PLOTTING COMPLETED")
print("=" * 60)
print(f"Plots created: {len(list(output_dir.glob('*.png')))}")
print(f"Output directory: {output_dir}")
print("Done!")
