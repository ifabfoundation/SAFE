#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=community_detection_G_NVG_p3_rms
#SBATCH --time=16:00:00
#SBATCH --cpus-per-task=18
#SBATCH --mem=320G
#SBATCH --output=logs/comm_detection_G_NVG_p3_rms_%j.out
#SBATCH --error=logs/comm_detection_G_NVG_p3_rms_%j.err
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
python src/anomaly_detection/graph_based/communities/comm_det_computations/community_detection.py 

echo "Training completato: $(date)"