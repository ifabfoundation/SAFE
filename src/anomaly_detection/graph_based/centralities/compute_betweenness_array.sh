#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=betweenness_array
#SBATCH --array=0-7
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --output=logs/betweenness_array_%A_%a.out
#SBATCH --error=logs/betweenness_array_%A_%a.err
#SBATCH --account=safe

# Array solo per p3_rms, p3_temp, p4_rms, p4_temp (NVG e HVG)
GRAPHS=(
    "G_NVG_p3_rms"
    "G_NVG_p3_temp"
    "G_NVG_p4_rms"
    "G_NVG_p4_temp"
    "G_HVG_p3_rms"
    "G_HVG_p3_temp"
    "G_HVG_p4_rms"
    "G_HVG_p4_temp"
)

# Seleziona il grafo per questo task
GRAPH_NAME=${GRAPHS[$SLURM_ARRAY_TASK_ID]}

# Informazioni sul job
echo "=== JOB INFO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Graph: $GRAPH_NAME"
echo "nSamples: 1000"
echo "Start Time: $(date)"
echo "================"

# Setup ambiente
cd /home/projects/safe/
eval "$(micromamba shell hook --shell bash)"
micromamba activate ./.mamba-env

# Esecuzione calcolo betweenness per singolo grafo
python src/anomaly_detection/graph_based/centralities/betweenness_single.py "$GRAPH_NAME"

echo "Computation completed: $(date)"
