
#plot of cliquestar oddball scores on Tampieri time series data
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_scores')
timeseries_file = Path('/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/oddball_vs_timeseries/cliquestar')
output_dir.mkdir(parents=True, exist_ok=True)


# Load time series data
df_ts = pd.read_csv(timeseries_file)

# Mapping between graph names and time series columns
def get_timeseries_column(graph_name):
    """Extract time series column name from graph name"""
    # Example: G_NVG_p3_rms -> bonfi/gb1_p3_acc_rms
    parts = graph_name.split('_')
    if len(parts) >= 4:
        sensor = parts[2]  # p3 or p4
        signal = '_'.join(parts[3:])  # rms, cfa, kurt, max, min, skew, std, temp
        
        # Map signal name to full column name
        if signal == 'temp':
            return f"bonfi/gb1_{sensor}_{signal}"
        else:
            return f"bonfi/gb1_{sensor}_acc_{signal}"
    return None

# Load oddball results
oddball_pickle = oddball_dir / "oddball_results.pkl"
print(f"\nLoading {oddball_pickle}...")
with open(oddball_pickle, 'rb') as f:
    oddball_results = pickle.load(f)

print(f"Loaded oddball results for {len(oddball_results)} graphs")

# Process each graph
for graph_name, oddball_df in oddball_results.items():
    try:
        # Get corresponding time series column
        ts_column = get_timeseries_column(graph_name)
        
        if ts_column not in df_ts.columns:
            print(f"Time series column '{ts_column}' not found for {graph_name}")
            continue
        
        # Get time series data
        ts_data = df_ts[ts_column].values
        
        # Get top 40 oddball scores
        top_oddball = oddball_df.nlargest(40, 'oddball_score')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Plot time series in gray
        ax.plot(range(len(ts_data)), ts_data, color='dimgray', linewidth=0.8, alpha=0.7, label='Time series')
        
        # Highlight top oddball points with colormap
        top_nodes = top_oddball['node'].values.astype(int)
        top_scores = top_oddball['oddball_score'].values
        
        # Filter valid indices
        valid_mask = top_nodes < len(ts_data)
        top_nodes = top_nodes[valid_mask]
        top_scores = top_scores[valid_mask]
        
        scatter = ax.scatter(top_nodes, ts_data[top_nodes], 
                            c=top_scores, cmap='hot_r', s=80, alpha=0.9,
                            edgecolors='black', linewidth=0.5,
                            label='Top 40 oddball nodes', zorder=5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Oddball Score', rotation=270, labelpad=20, fontsize=11)
        
        ax.set_xlabel('Time Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Top 40 Oddball Nodes - {graph_name}\n(Cliquestar: $E_i \\sim N_i^\\alpha$)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Save plot
        plot_file = output_dir / f"{graph_name}_oddball_on_timeseries.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved: {plot_file.name}")
        
    except Exception as e:
        print(f"Error processing {graph_name}: {e}")
        continue

print(f"\nPlots created: {len(list(output_dir.glob('*_oddball_on_timeseries.png')))} files")
print(f"Output directory: {output_dir}")
print("Done!")
