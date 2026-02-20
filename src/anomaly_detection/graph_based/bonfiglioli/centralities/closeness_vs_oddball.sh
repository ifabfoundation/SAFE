#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=closeness_vs_oddball
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --output=logs/closeness_vs_oddball_%j.out
#SBATCH --error=logs/closeness_vs_oddball_%j.err
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
python src/anomaly_detection/graph_based/bonfiglioli/centralities/closeness_vs_oddball.py 

echo "Closeness vs Oddball plotting completed: $(date)"