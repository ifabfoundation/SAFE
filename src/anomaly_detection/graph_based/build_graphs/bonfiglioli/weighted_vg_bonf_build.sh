#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=weighted_vg_bonf_build
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=14
#SBATCH --mem=250G
#SBATCH --output=logs/weighted_vg_bonf_build_and_degrees_%j.out
#SBATCH --error=logs/weighted_vg_bonf_build_and_degrees_%j.err
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
python src/anomaly_detection/graph_based/weighted/weighted_vg_bonf_build.py 

echo "Training completato: $(date)"