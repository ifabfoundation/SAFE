#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=oddball_lambda
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=200G
#SBATCH --output=logs/oddball_lambda_%j.out
#SBATCH --error=logs/oddball_lambda_%j.err
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
python src/anomaly_detection/graph_based/oddball/oddball_lambda.py 

echo "Training completato: $(date)"