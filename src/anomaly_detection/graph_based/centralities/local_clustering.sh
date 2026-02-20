#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=compute_local_clustering
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=150G
#SBATCH --output=logs/compute_local_clustering_%j.out
#SBATCH --error=logs/compute_local_clustering_%j.err
#SBATCH --account=safe

# Informazioni sul job
echo "=== JOB INFO ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "================"

# Setup ambiente
cd /home/projects/safe/
eval "$(micromamba shell hook --shell bash)"
micromamba activate ./.mamba-env

# Esecuzione training
python src/anomaly_detection/graph_based/centralities/local_clustering.py 

echo "Training completato: $(date)"