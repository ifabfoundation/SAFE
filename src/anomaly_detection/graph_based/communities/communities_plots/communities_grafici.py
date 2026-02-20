# Plot time series with community detection overlay
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

print("COMMUNITY DETECTION VISUALIZATION")
print("=" * 60)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Paths
communities_dir = Path('/home/projects/safe/outputs/networks/communities')
output_dir = Path('/home/projects/safe/outputs/networks/communities/plots')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Communities directory: {communities_dir}")
print(f"Output directory: {output_dir}")

# Load community results
print("\nLoading community results...")
pickle_file = communities_dir / "community_results.pkl"

try:
    with open(pickle_file, 'rb') as f:
        community_results = pickle.load(f)
    print(f"Loaded community data for {len(community_results)} graphs")
except FileNotFoundError:
    print(f"Error: {pickle_file} not found")
    print("Please run community_detection.py first")
    exit(1)

# Load original time series data
print("\nLoading time series data...")
ts_file = Path('/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv')
df_full = pd.read_csv(ts_file, index_col=0)
print(f"Loaded time series: {df_full.shape[0]} rows, {df_full.shape[1]} columns")

# Define time series mapping (graph name -> column name)
series_mapping = {
    # P3 sensor
    'G_NVG_p3_rms': 'bonfi/gb1_p3_acc_rms',
    'G_HVG_p3_rms': 'bonfi/gb1_p3_acc_rms',
    'G_NVG_p3_cfa': 'bonfi/gb1_p3_acc_cfa',
    'G_HVG_p3_cfa': 'bonfi/gb1_p3_acc_cfa',
    'G_NVG_p3_kurt': 'bonfi/gb1_p3_acc_kurt',
    'G_HVG_p3_kurt': 'bonfi/gb1_p3_acc_kurt',
    'G_NVG_p3_max': 'bonfi/gb1_p3_acc_max',
    'G_HVG_p3_max': 'bonfi/gb1_p3_acc_max',
    'G_NVG_p3_min': 'bonfi/gb1_p3_acc_min',
    'G_HVG_p3_min': 'bonfi/gb1_p3_acc_min',
    'G_NVG_p3_skew': 'bonfi/gb1_p3_acc_skew',
    'G_HVG_p3_skew': 'bonfi/gb1_p3_acc_skew',
    'G_NVG_p3_std': 'bonfi/gb1_p3_acc_std',
    'G_HVG_p3_std': 'bonfi/gb1_p3_acc_std',
    'G_NVG_p3_temp': 'bonfi/gb1_p3_temp',
    'G_HVG_p3_temp': 'bonfi/gb1_p3_temp',
    # P4 sensor
    'G_NVG_p4_rms': 'bonfi/gb1_p4_acc_rms',
    'G_HVG_p4_rms': 'bonfi/gb1_p4_acc_rms',
    'G_NVG_p4_cfa': 'bonfi/gb1_p4_acc_cfa',
    'G_HVG_p4_cfa': 'bonfi/gb1_p4_acc_cfa',
    'G_NVG_p4_kurt': 'bonfi/gb1_p4_acc_kurt',
    'G_HVG_p4_kurt': 'bonfi/gb1_p4_acc_kurt',
    'G_NVG_p4_max': 'bonfi/gb1_p4_acc_max',
    'G_HVG_p4_max': 'bonfi/gb1_p4_acc_max',
    'G_NVG_p4_min': 'bonfi/gb1_p4_acc_min',
    'G_HVG_p4_min': 'bonfi/gb1_p4_acc_min',
    'G_NVG_p4_skew': 'bonfi/gb1_p4_acc_skew',
    'G_HVG_p4_skew': 'bonfi/gb1_p4_acc_skew',
    'G_NVG_p4_std': 'bonfi/gb1_p4_acc_std',
    'G_HVG_p4_std': 'bonfi/gb1_p4_acc_std',
    'G_NVG_p4_temp': 'bonfi/gb1_p4_temp',
    'G_HVG_p4_temp': 'bonfi/gb1_p4_temp',
}

def plot_timeseries_with_communities(graph_name, community_df, time_series, output_dir):
    """
    Plot time series with communities highlighted in different colors.
    """
    print(f"Plotting {graph_name}...")
    
    # Get unique communities
    communities = sorted(community_df['community'].unique())
    num_communities = len(communities)
    
    # Generate colors for communities
    colors = plt.cm.tab20(np.linspace(0, 1, num_communities))
    community_colors = {comm: colors[i] for i, comm in enumerate(communities)}
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # Plot 1: Time series with community colors
    time_index = np.arange(len(time_series))
    
    # Plot each segment by community
    for node_idx in range(len(time_series)):
        if node_idx < len(community_df):
            comm = community_df.iloc[node_idx]['community']
            color = community_colors.get(comm, 'gray')
            
            # Plot single point with community color
            if node_idx < len(time_series) - 1:
                ax1.plot([node_idx, node_idx + 1], 
                        [time_series.iloc[node_idx], time_series.iloc[node_idx + 1]],
                        color=color, linewidth=1.5, alpha=0.8)
    
    ax1.set_ylabel('Value')
    ax1.set_title(f'{graph_name} - Time Series with Communities', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Community assignment over time
    community_sequence = []
    for node_idx in range(len(time_series)):
        if node_idx < len(community_df):
            comm = community_df.iloc[node_idx]['community']
            community_sequence.append(comm)
        else:
            community_sequence.append(-1)
    
    # Create colored bands for communities
    for node_idx in range(len(community_sequence)):
        comm = community_sequence[node_idx]
        color = community_colors.get(comm, 'gray')
        ax2.axvspan(node_idx, node_idx + 1, facecolor=color, alpha=0.7)
    
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Community')
    ax2.set_title('Community Assignment Over Time', fontsize=12, fontweight='bold')
    ax2.set_ylim(-0.5, num_communities + 0.5)
    ax2.set_yticks(range(num_communities))
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add statistics text
    stats_text = f"""Communities: {num_communities}
Nodes: {len(community_df)}
Time points: {len(time_series)}
Modularity: {community_df['modularity'].iloc[0]:.4f if 'modularity' in community_df.columns else 'N/A'}"""
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='white', alpha=0.9), fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{graph_name}_communities_timeseries.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

# Create plots for all graphs
print("\nCreating community visualization plots...")
print("=" * 60)

plot_count = 0

for graph_name, community_df in community_results.items():
    try:
        # Get corresponding time series column
        if graph_name not in series_mapping:
            print(f"Warning: No time series mapping for {graph_name}, skipping")
            continue
        
        ts_column = series_mapping[graph_name]
        
        if ts_column not in df_full.columns:
            print(f"Warning: Column {ts_column} not found in dataset, skipping")
            continue
        
        # Get time series (use same length as community data)
        time_series = df_full[ts_column].iloc[:len(community_df)]
        
        # Create plot
        plot_file = plot_timeseries_with_communities(graph_name, community_df, time_series, output_dir)
        print(f"Saved: {plot_file.name}")
        plot_count += 1
        
    except Exception as e:
        print(f"Error plotting {graph_name}: {e}")
        continue

print(f"\n{'='*60}")
print(f"VISUALIZATION COMPLETED")
print(f"Plots created: {plot_count}/{len(community_results)}")
print(f"Output directory: {output_dir}")
print("All community plots saved successfully!")
