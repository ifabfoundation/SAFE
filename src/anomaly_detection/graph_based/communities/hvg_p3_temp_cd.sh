#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=hvg_p3_temp_cd
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --output=logs/hvg_p3_temp_cd_%j.out
#SBATCH --error=logs/hvg_p3_temp_cd_%j.err
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
python src/anomaly_detection/graph_based/communities/hvg_p3_temp_cd.py 

echo "Training completato: $(date)"