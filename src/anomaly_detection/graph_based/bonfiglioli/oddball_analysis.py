# Analysis of Weighted vs Lambda Oddball Scores for Bonfiglioli Graphs

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Paths
weighted_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores/weighted_scores')
lambda_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/oddball_scores/lambda_scores')
output_dir = Path('/home/projects/safe/outputs/networks/oddball/bonfiglioli/analysis')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYSIS: Weighted vs Lambda Oddball Scores")
print("=" * 80)

# Load files from both directories
weighted_files = {f.stem.replace('_oddball_weighted_results', ''): f for f in weighted_dir.glob('*_oddball_weighted_results.csv')}
lambda_files = {f.stem.replace('_oddball_lambda_results', ''): f for f in lambda_dir.glob('*_oddball_lambda_results.csv')}

print(f"\nWeighted files: {len(weighted_files)}")
print(f"Lambda files: {len(lambda_files)}")

# Find common graphs
common_graphs = set(weighted_files.keys()) & set(lambda_files.keys())
print(f"\nCommon graphs: {len(common_graphs)}")
print(f"Graphs: {sorted(common_graphs)}")

# Analysis results
results = []

print("\n" + "=" * 80)
print("COMPARING TOP 100 NODES")
print("=" * 80)

for graph_name in sorted(common_graphs):
    print(f"\n{graph_name}")
    print("-" * 80)
    
    # Load both datasets
    df_weighted = pd.read_csv(weighted_files[graph_name])
    df_lambda = pd.read_csv(lambda_files[graph_name])
    
    # Get top 100 nodes from each
    top100_weighted = set(df_weighted.nlargest(100, 'oddball_score')['node'].values)
    top100_lambda = set(df_lambda.nlargest(100, 'oddball_score')['node'].values)
    
    # Calculate overlap
    common_nodes = top100_weighted & top100_lambda
    overlap_count = len(common_nodes)
    overlap_pct = (overlap_count / 100) * 100
    
    # Only in weighted
    only_weighted = top100_weighted - top100_lambda
    only_lambda = top100_lambda - top100_weighted
    
    print(f"  Top 100 overlap: {overlap_count} nodes ({overlap_pct:.1f}%)")
    print(f"  Only in weighted: {len(only_weighted)} nodes")
    print(f"  Only in lambda: {len(only_lambda)} nodes")
    print(f"  Common nodes (first 20): {sorted(list(common_nodes))[:20]}")
    
    # Analyze "zones" - check if common nodes are clustered
    if common_nodes:
        common_sorted = sorted(list(common_nodes))
        zones = []
        current_zone = [common_sorted[0]]
        
        for i in range(1, len(common_sorted)):
            if common_sorted[i] - common_sorted[i-1] <= 100:  # Consider nodes within 100 indices as same zone
                current_zone.append(common_sorted[i])
            else:
                if len(current_zone) >= 3:  # Only consider zones with at least 3 nodes
                    zones.append((current_zone[0], current_zone[-1], len(current_zone)))
                current_zone = [common_sorted[i]]
        
        # Add last zone
        if len(current_zone) >= 3:
            zones.append((current_zone[0], current_zone[-1], len(current_zone)))
        
        print(f"  Zones found: {len(zones)}")
        for start, end, count in zones:
            print(f"    Zone: nodes {start}-{end} ({count} common nodes)")
    else:
        zones = []
    
    # Store results
    results.append({
        'graph': graph_name,
        'overlap_count': overlap_count,
        'overlap_pct': overlap_pct,
        'only_weighted': len(only_weighted),
        'only_lambda': len(only_lambda),
        'num_zones': len(zones),
        'common_nodes': sorted(list(common_nodes))
    })

# Create summary DataFrame
df_results = pd.DataFrame(results)

# Check if we have results
if len(df_results) == 0:
    print("\n" + "=" * 80)
    print("NO COMMON GRAPHS FOUND")
    print("=" * 80)
    print("No graphs were found in both weighted and lambda directories.")
    print("Cannot perform comparison analysis.")
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE (NO RESULTS)")
    print("=" * 80)
    exit(0)

print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print(f"\nOverlap statistics:")
print(f"  Average overlap: {df_results['overlap_pct'].mean():.1f}%")
print(f"  Min overlap: {df_results['overlap_pct'].min():.1f}%")
print(f"  Max overlap: {df_results['overlap_pct'].max():.1f}%")
print(f"  Std deviation: {df_results['overlap_pct'].std():.1f}%")
print(f"\nZone statistics:")
print(f"  Average number of zones: {df_results['num_zones'].mean():.1f}")
print(f"  Max zones in a graph: {df_results['num_zones'].max()}")

# Save detailed results to CSV
results_file = output_dir / 'weighted_vs_lambda_comparison.csv'
df_results[['graph', 'overlap_count', 'overlap_pct', 'only_weighted', 'only_lambda', 'num_zones']].to_csv(results_file, index=False)
print(f"\nResults saved to: {results_file}")

# Save common nodes for each graph
for idx, row in df_results.iterrows():
    if row['common_nodes']:
        nodes_file = output_dir / f"{row['graph']}_common_nodes.txt"
        with open(nodes_file, 'w') as f:
            f.write(f"Common nodes between weighted and lambda for {row['graph']}\n")
            f.write(f"Total: {len(row['common_nodes'])} nodes\n\n")
            f.write("Node indices:\n")
            for node in row['common_nodes']:
                f.write(f"{node}\n")

print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

# Create visualization: overlap percentage per graph
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(df_results)), df_results['overlap_pct'], color='steelblue', alpha=0.7)
ax.set_xticks(range(len(df_results)))
ax.set_xticklabels(df_results['graph'], rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Graph', fontsize=12)
ax.set_ylabel('Overlap (%)', fontsize=12)
ax.set_title('Top 100 Nodes Overlap: Weighted vs Lambda Oddball', fontsize=14, fontweight='bold')
ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% overlap')
ax.axhline(y=df_results['overlap_pct'].mean(), color='green', linestyle='--', alpha=0.5, label=f"Mean: {df_results['overlap_pct'].mean():.1f}%")

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, df_results['overlap_pct'])):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
            f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(output_dir / 'overlap_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: overlap_comparison.png")

# Create visualization: number of zones per graph
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(len(df_results)), df_results['num_zones'], color='coral', alpha=0.7)
ax.set_xticks(range(len(df_results)))
ax.set_xticklabels(df_results['graph'], rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Graph', fontsize=12)
ax.set_ylabel('Number of Zones', fontsize=12)
ax.set_title('Number of Clustered Zones in Common Nodes (within 100 indices)', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, df_results['num_zones'])):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{int(val)}', ha='center', va='bottom', fontsize=9)

ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'zones_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: zones_comparison.png")

# Create stacked bar chart: overlap vs unique nodes
fig, ax = plt.subplots(figsize=(14, 6))
x_pos = range(len(df_results))
p1 = ax.bar(x_pos, df_results['overlap_count'], label='Common nodes', color='green', alpha=0.7)
p2 = ax.bar(x_pos, df_results['only_weighted'], bottom=df_results['overlap_count'], 
            label='Only in Weighted', color='blue', alpha=0.7)
p3 = ax.bar(x_pos, df_results['only_lambda'], bottom=df_results['overlap_count'] + df_results['only_weighted'],
            label='Only in Lambda', color='orange', alpha=0.7)

ax.set_xticks(x_pos)
ax.set_xticklabels(df_results['graph'], rotation=45, ha='right', fontsize=9)
ax.set_xlabel('Graph', fontsize=12)
ax.set_ylabel('Number of Nodes', fontsize=12)
ax.set_title('Distribution of Top 100 Nodes: Weighted vs Lambda', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'distribution_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: distribution_comparison.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"Output directory: {output_dir}")

