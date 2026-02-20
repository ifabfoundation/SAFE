# Plot Oddball Score vs Degree for Bonfiglioli Graphs

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
oddball_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores')
degree_dir = Path('/home/projects/safe/outputs/networks/grafi/bonfiglioli_graphs/degrees')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_vs_degree')
output_dir.mkdir(parents=True, exist_ok=True)

print("Finding oddball CSV files...")
print(f"Oddball directory: {oddball_dir}")
print(f"Degree directory: {degree_dir}")

# Find all oddball CSV files
oddball_files = list(oddball_dir.glob('*_oddball_results.csv'))
print(f"Found {len(oddball_files)} oddball files")

# Extract graph names
graph_names = [f.stem.replace('_oddball_results', '') for f in oddball_files]
print(f"Graphs: {graph_names}")

print("\n" + "=" * 60)
print("PLOTTING ODDBALL VS DEGREE")
print("=" * 60)

# Process each graph
for graph_name in graph_names:
    print(f"\nProcessing {graph_name}...")
    
    # Load oddball data
    oddball_file = oddball_dir / f"{graph_name}_oddball_results.csv"
    if not oddball_file.exists():
        print(f"  Warning: Oddball file not found: {oddball_file.name}")
        continue
    
    df_oddball = pd.read_csv(oddball_file)
    print(f"  Loaded oddball: {len(df_oddball)} nodes")
    
    # Load degree data
    degree_file = degree_dir / f"{graph_name}_degrees.csv"
    if not degree_file.exists():
        print(f"  Warning: Degree file not found: {degree_file.name}")
        continue
    
    df_degree = pd.read_csv(degree_file)
    print(f"  Loaded degrees: {len(df_degree)} nodes")
    
    # Merge on node index
    df_merged = pd.merge(df_oddball[['node', 'oddball_score']], df_degree, on='node', how='inner')
    
    if len(df_merged) == 0:
        print(f"  Warning: No matching nodes")
        continue
    
    print(f"  Merged {len(df_merged)} nodes")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df_merged['degree'], 
                        df_merged['oddball_score'],
                        alpha=0.6, 
                        s=30,
                        c=df_merged['node'],
                        cmap='viridis',
                        edgecolors='black',
                        linewidth=0.3)
    
    ax.set_xlabel('Degree', fontsize=12)
    ax.set_ylabel('Oddball Score', fontsize=12)
    ax.set_title(f'Oddball Score vs Degree - {graph_name}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Node Index\n(position in time series)', fontsize=11, labelpad=15)
    cbar.ax.tick_params(labelsize=9)
    
    # Save plot
    plot_file = output_dir / f"{graph_name}_oddball_vs_degree.png"
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
