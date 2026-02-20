#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=vg_reducted
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --output=logs/vg_reducted_%j.out
#SBATCH --error=logs/vg_reducted_%j.err
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
python src/anomaly_detection/graph_based/communities/vg_reducted.py 

echo "Training completato: $(date)"