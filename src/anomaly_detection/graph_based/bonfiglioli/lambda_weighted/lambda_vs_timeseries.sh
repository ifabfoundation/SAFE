#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=lambda_vs_timeseries_bonf
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/lambda_vs_timeseries_bonf_%j.out
#SBATCH --error=logs/lambda_vs_timeseries_bonf_%j.err
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
python src/anomaly_detection/graph_based/bonfiglioli/lambda_weighted/lambda_vs_timeseries.py 

echo "Degree calculation completed: $(date)"