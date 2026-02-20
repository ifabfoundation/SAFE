#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=analyze_branches
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=5G
#SBATCH --output=logs/analyze_branches_%j.out
#SBATCH --error=logs/analyze_branches_%j.err
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
python src/anomaly_detection/graph_based/bonfiglioli/oddball_weighted/analyze_branches.py 

echo "Training completato: $(date)"