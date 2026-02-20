# Import required libraries
import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import seaborn as sns
from collections import Counter

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300




# Load degree analysis results from network_analysis
import pickle
from pathlib import Path

# Path to degree analysis results
degree_dir = Path('/home/projects/safe/outputs/networks/degree')

print("Loading degree analysis results...")
print("=" * 50)

try:
    # Load degree distributions
    with open(degree_dir / 'degree_distributions.pkl', 'rb') as f:
        degree_distributions = pickle.load(f)
    print(f"✅ Loaded degree_distributions: {len(degree_distributions)} graphs")
    
    # Load average degrees
    with open(degree_dir / 'average_degrees.pkl', 'rb') as f:
        average_degrees = pickle.load(f)
    print(f"✅ Loaded average_degrees: {len(average_degrees)} graphs")
    
    # Verify data consistency
    if set(degree_distributions.keys()) == set(average_degrees.keys()):
        print("✅ Data consistency verified")
    else:
        print("⚠️ Warning: Mismatch between degree_distributions and average_degrees keys")
    
    # Show data summary
    nvg_count = sum(1 for data in degree_distributions.values() if data['graph_type'] == 'NVG')
    hvg_count = sum(1 for data in degree_distributions.values() if data['graph_type'] == 'HVG')
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"  - Total graphs: {len(degree_distributions)}")
    print(f"  - NVG graphs: {nvg_count}")
    print(f"  - HVG graphs: {hvg_count}")
    
    print(f"\n📋 Available graphs:")
    for name, data in list(degree_distributions.items())[:5]:
        graph_type = data['graph_type']
        avg_deg = average_degrees[name]
        print(f"  - {name}: {graph_type}, avg_degree={avg_deg:.4f}")
    if len(degree_distributions) > 5:
        print(f"  ... and {len(degree_distributions) - 5} more")

except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    print("🔄 Make sure to run network_analysis first to generate the data.")
except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()





    # Create degree distribution plots for each graph
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Create output directory for plots
plots_dir = Path('/home/projects/safe/outputs/networks/degree/degrees_plots')
plots_dir.mkdir(parents=True, exist_ok=True)

print("Creating degree distribution plots...")
print(f"Output directory: {plots_dir}")
print("=" * 60)

def plot_degree_distribution(name, data, avg_degree, save_dir):
    """
    Create and save degree distribution plot for a single graph.
    """
    degrees = data['degrees']
    degree_counts = data['degree_counts']
    graph_type = data['graph_type']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Histogram of degrees
    ax1.hist(degrees, bins=50, alpha=0.7, edgecolor='black', 
             color='skyblue' if graph_type == 'NVG' else 'lightcoral')
    ax1.axvline(avg_degree, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {avg_degree:.2f}')
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Number of Nodes')
    ax1.set_title(f'Degree Distribution - {name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log plot of degree distribution
    degrees_sorted = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees_sorted]
    
    ax2.loglog(degrees_sorted, counts, 'o-', markersize=4, alpha=0.7,
               color='navy' if graph_type == 'NVG' else 'darkred')
    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('Count (log scale)')
    ax2.set_title(f'Log-Log Degree Distribution - {name}')
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f"""Graph: {graph_type}
Nodes: {len(degrees):,}
Min degree: {min(degrees)}
Max degree: {max(degrees)}
Mean degree: {avg_degree:.4f}
Std degree: {np.std(degrees):.4f}"""
    
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', 
             facecolor='white', alpha=0.8), fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{name}_degree_distribution.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()
    
    return filepath

# Create plots for all graphs
saved_plots = []
total_graphs = len(degree_distributions)

print(f"Creating plots for {total_graphs} graphs...\n")

for i, (name, data) in enumerate(degree_distributions.items(), 1):
    avg_deg = average_degrees[name]
    graph_type = data['graph_type']
    
    print(f"[{i:2d}/{total_graphs}] Processing {name} ({graph_type})...")
    
    try:
        filepath = plot_degree_distribution(name, data, avg_deg, plots_dir)
        saved_plots.append(filepath.name)
        print(f"         ✅ Saved: {filepath.name}")
        
    except Exception as e:
        print(f"         ❌ Error: {e}")
        continue

print(f"\n{'='*60}")
print(f"📊 PLOTTING COMPLETED")
print(f"📁 Location: {plots_dir}")
print(f"📈 Plots created: {len(saved_plots)}/{total_graphs}")
print(f"💾 Total files: {len(list(plots_dir.glob('*.png')))} PNG files")

if len(saved_plots) < total_graphs:
    failed = total_graphs - len(saved_plots)
    print(f"⚠️  {failed} plots failed to generate")






    # Create summary comparison plots
import matplotlib.pyplot as plt
import numpy as np

print("Creating summary comparison plots...")

# Separate data by graph type
nvg_data = {name: data for name, data in degree_distributions.items() if data['graph_type'] == 'NVG'}
hvg_data = {name: data for name, data in degree_distributions.items() if data['graph_type'] == 'HVG'}

# Create comprehensive comparison plot
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# Plot 1: Average degrees comparison
nvg_avg = [average_degrees[name] for name in nvg_data.keys()]
hvg_avg = [average_degrees[name] for name in hvg_data.keys()]

axes[0, 0].boxplot([nvg_avg, hvg_avg], labels=['NVG', 'HVG'])
axes[0, 0].set_title('Average Degree Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Average Degree')
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Degree ranges comparison
nvg_ranges = [(min(data['degrees']), max(data['degrees'])) for data in nvg_data.values()]
hvg_ranges = [(min(data['degrees']), max(data['degrees'])) for data in hvg_data.values()]

nvg_mins, nvg_maxs = zip(*nvg_ranges)
hvg_mins, hvg_maxs = zip(*hvg_ranges)

x_nvg = np.ones(len(nvg_mins)) * 0.8 + np.random.normal(0, 0.05, len(nvg_mins))
x_hvg = np.ones(len(hvg_mins)) * 2.2 + np.random.normal(0, 0.05, len(hvg_mins))

axes[0, 1].scatter(x_nvg, nvg_mins, alpha=0.6, label='NVG Min', color='blue')
axes[0, 1].scatter(x_nvg, nvg_maxs, alpha=0.6, label='NVG Max', color='lightblue')
axes[0, 1].scatter(x_hvg, hvg_mins, alpha=0.6, label='HVG Min', color='red')
axes[0, 1].scatter(x_hvg, hvg_maxs, alpha=0.6, label='HVG Max', color='lightcoral')

axes[0, 1].set_xlim(0, 3)
axes[0, 1].set_xticks([1, 2])
axes[0, 1].set_xticklabels(['NVG', 'HVG'])
axes[0, 1].set_ylabel('Degree')
axes[0, 1].set_title('Degree Ranges Comparison', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Number of nodes comparison
nvg_nodes = [len(data['degrees']) for data in nvg_data.values()]
hvg_nodes = [len(data['degrees']) for data in hvg_data.values()]

axes[0, 2].boxplot([nvg_nodes, hvg_nodes], labels=['NVG', 'HVG'])
axes[0, 2].set_title('Number of Nodes Comparison', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('Number of Nodes')
axes[0, 2].grid(True, alpha=0.3)

# Plot 4: P3 vs P4 sensor comparison
p3_avg = [average_degrees[name] for name in average_degrees.keys() if 'p3' in name]
p4_avg = [average_degrees[name] for name in average_degrees.keys() if 'p4' in name]

axes[1, 0].boxplot([p3_avg, p4_avg], labels=['P3 Sensor', 'P4 Sensor'])
axes[1, 0].set_title('Sensor Comparison (Average Degree)', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Degree')
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Series type comparison
series_types = ['rms', 'cfa', 'kurt', 'max', 'min', 'skew', 'std', 'temp']
series_avg_degrees = []
series_labels = []

for series in series_types:
    series_avgs = [average_degrees[name] for name in average_degrees.keys() if series in name]
    if series_avgs:
        series_avg_degrees.extend(series_avgs)
        series_labels.extend([series] * len(series_avgs))

# Create a boxplot for series
unique_series = list(set(series_labels))
series_data = [[average_degrees[name] for name in average_degrees.keys() if series in name] 
               for series in unique_series]

axes[1, 1].boxplot(series_data, labels=unique_series)
axes[1, 1].set_title('Average Degree by Series Type', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Average Degree')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Overall statistics
all_degrees = []
for data in degree_distributions.values():
    all_degrees.extend(data['degrees'])

axes[1, 2].hist(all_degrees, bins=100, alpha=0.7, color='purple', edgecolor='black')
axes[1, 2].set_xlabel('Degree')
axes[1, 2].set_ylabel('Count (All Graphs)')
axes[1, 2].set_title('Overall Degree Distribution', fontsize=14, fontweight='bold')
axes[1, 2].set_yscale('log')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()

# Save summary plot
summary_filepath = plots_dir / "degree_distributions_summary.png"
plt.savefig(summary_filepath, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Summary plot saved: {summary_filepath.name}")

# Print summary statistics
print(f"\n📊 SUMMARY STATISTICS")
print(f"="*50)
print(f"NVG graphs: {len(nvg_data)}")
print(f"  - Average degree range: {min(nvg_avg):.4f} - {max(nvg_avg):.4f}")
print(f"  - Mean average degree: {np.mean(nvg_avg):.4f} ± {np.std(nvg_avg):.4f}")

print(f"\nHVG graphs: {len(hvg_data)}")
print(f"  - Average degree range: {min(hvg_avg):.4f} - {max(hvg_avg):.4f}")
print(f"  - Mean average degree: {np.mean(hvg_avg):.4f} ± {np.std(hvg_avg):.4f}")

degree_reduction = ((np.mean(nvg_avg) - np.mean(hvg_avg)) / np.mean(nvg_avg)) * 100
print(f"\nDegree reduction (NVG→HVG): {degree_reduction:.1f}%")

print(f"\nSensor comparison:")
print(f"  - P3 average degree: {np.mean(p3_avg):.4f} ± {np.std(p3_avg):.4f}")
print(f"  - P4 average degree: {np.mean(p4_avg):.4f} ± {np.std(p4_avg):.4f}")


# === SAVE ALL OPEN FIGURES USING THEIR TITLES ===
import matplotlib.pyplot as plt
from pathlib import Path
import re

save_dir = Path('/home/projects/safe/outputs/networks/degree/degrees_plots')
save_dir.mkdir(parents=True, exist_ok=True)

print("\n🔄 Saving all open figures with title-based filenames...")

def sanitize(name):
    """Remove characters not allowed in filenames."""
    name = name.lower().strip()
    name = re.sub(r'[^a-z0-9_\-]+', '_', name)
    return name

for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    
    # Try to extract title from the first axes
    axes = fig.get_axes()
    if axes:
        title = axes[0].get_title()
    else:
        title = ""

    if title:
        filename = sanitize(title) + ".png"
    else:
        filename = f"figure_{fig_num}.png"

    outfile = save_dir / filename
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {outfile.name}")

print("📁 All open figures saved with title-based names.")
