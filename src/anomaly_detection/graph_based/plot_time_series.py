# Plot time series from processed dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Create output directory
output_dir = Path('/home/projects/safe/outputs/networks/plot_time_series')
output_dir.mkdir(parents=True, exist_ok=True)

print("Loading and plotting time series...")
print("=" * 50)

# Load the dataset
df = pd.read_csv('/home/projects/safe/data/processed-datasets/processed_streaming_row_continuous.csv', index_col=0)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")


# Define time series
# P3 sensor series
df1 = df['bonfi/gb1_p3_acc_rms']
df2 = df['bonfi/gb1_p3_acc_cfa']
df3 = df['bonfi/gb1_p3_acc_kurt']
df4 = df['bonfi/gb1_p3_acc_max']
df5 = df['bonfi/gb1_p3_acc_min']
df6 = df['bonfi/gb1_p3_acc_skew']
df7 = df['bonfi/gb1_p3_acc_std']
df8 = df['bonfi/gb1_p3_temp']

# P4 sensor series
df1_4 = df['bonfi/gb1_p4_acc_rms']
df2_4 = df['bonfi/gb1_p4_acc_cfa']
df3_4 = df['bonfi/gb1_p4_acc_kurt']
df4_4 = df['bonfi/gb1_p4_acc_max']
df5_4 = df['bonfi/gb1_p4_acc_min']
df6_4 = df['bonfi/gb1_p4_acc_skew']
df7_4 = df['bonfi/gb1_p4_acc_std']
df8_4 = df['bonfi/gb1_p4_temp']

# Organize series data
p3_series = [
    (df1, 'p3_rms', 'RMS'),
    (df2, 'p3_cfa', 'CFA'), 
    (df3, 'p3_kurt', 'Kurtosis'),
    (df4, 'p3_max', 'Max'),
    (df5, 'p3_min', 'Min'),
    (df6, 'p3_skew', 'Skewness'),
    (df7, 'p3_std', 'Std Dev'),
    (df8, 'p3_temp', 'Temperature')
]

p4_series = [
    (df1_4, 'p4_rms', 'RMS'),
    (df2_4, 'p4_cfa', 'CFA'),
    (df3_4, 'p4_kurt', 'Kurtosis'),
    (df4_4, 'p4_max', 'Max'),
    (df5_4, 'p4_min', 'Min'),
    (df6_4, 'p4_skew', 'Skewness'),
    (df7_4, 'p4_std', 'Std Dev'),
    (df8_4, 'p4_temp', 'Temperature')
]

all_series = p3_series + p4_series

# Function to create individual time series plot
def plot_individual_series(series_data, series_name, series_title, sensor, save_dir):
    """Plot individual time series and save"""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Create time index (assuming sequential time points)
    time_index = range(len(series_data))
    
    # Plot the series
    color = 'blue' if 'p3' in sensor else 'red'
    ax.plot(time_index, series_data.values, color=color, linewidth=0.8, alpha=0.8)
    
    # Formatting
    ax.set_title(f'{sensor.upper()} Sensor - {series_title} Time Series', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Time Points')
    ax.set_ylabel(f'{series_title}')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{series_name}_time_series.png"
    filepath = save_dir / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    return filepath

# Plot individual series
print("\nCreating individual time series plots...")
saved_plots = []

for i, (series_data, series_name, series_title) in enumerate(all_series, 1):
    sensor = 'p3' if 'p3' in series_name else 'p4'
    
    print(f"[{i:2d}/16] Plotting {series_name}...")
    
    try:
        filepath = plot_individual_series(series_data, series_name, series_title, sensor, output_dir)
        saved_plots.append(filepath.name)
        print(f"         ✅ Saved: {filepath.name}")
        
    except Exception as e:
        print(f"         ❌ Error: {e}")
        continue

# Summary
print(f"\n{'='*60}")
print(f"📊 TIME SERIES PLOTTING COMPLETED")
print(f"📁 Output directory: {output_dir}")
print(f"📈 Individual plots: {len(saved_plots)}/16")
print(f"✅ All time series plots saved successfully!")

# List created files
print(f"\n📄 Files created:")
for png_file in sorted(output_dir.glob("*.png")):
    print(f"  - {png_file.name}")

plt.tight_layout()
comparison_file = output_dir / "p3_vs_p4_comparison.png"
plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {comparison_file.name}")

# 4. Overall summary plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))

# Plot all P3 series normalized
ax1.set_title('P3 Sensor - All Series (Normalized)', fontsize=14, fontweight='bold')
for i, (series_data, series_name, series_title) in enumerate(p3_series):
    normalized_data = (series_data - series_data.mean()) / series_data.std()
    time_index = range(len(normalized_data))
    ax1.plot(time_index, normalized_data.values, linewidth=1, alpha=0.7, label=series_title)

ax1.set_xlabel('Time Points')
ax1.set_ylabel('Normalized Values')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot all P4 series normalized
ax2.set_title('P4 Sensor - All Series (Normalized)', fontsize=14, fontweight='bold')
for i, (series_data, series_name, series_title) in enumerate(p4_series):
    normalized_data = (series_data - series_data.mean()) / series_data.std()
    time_index = range(len(normalized_data))
    ax2.plot(time_index, normalized_data.values, linewidth=1, alpha=0.7, label=series_title)

ax2.set_xlabel('Time Points')
ax2.set_ylabel('Normalized Values')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
summary_file = output_dir / "all_series_normalized_summary.png"
plt.savefig(summary_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {summary_file.name}")

# Summary
total_files = len(list(output_dir.glob("*.png")))
print(f"\n{'='*60}")
print(f"📊 TIME SERIES PLOTTING COMPLETED")
print(f"📁 Output directory: {output_dir}")
print(f"📈 Individual plots: {len(saved_plots)}/16")
print(f"📈 Combined plots: 4 files")
print(f"💾 Total files: {total_files} PNG files")
print(f"✅ All time series plots saved successfully!")

# List all created files
print(f"\n📄 Files created:")
for png_file in sorted(output_dir.glob("*.png")):
    print(f"  - {png_file.name}")
