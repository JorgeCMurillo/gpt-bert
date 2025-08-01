#!/bin/bash

#SBATCH --job-name=BERT
#SBATCH --account=XXX
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=7
#SBATCH --mem=0
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --partition=XXX
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --output=bert-%j.out


# This is an example script for running the BERT model pretraining on a single node with 8 GPUs.
# You'll most likely have to adjust the script to match your setup.

# distributed setup
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=9999
export WORLD_SIZE=$SLURM_NTASKS

CONTAINER="CONTAINER_LOCATION"
SING_BIND="SINGULARITY_BINDING"

set -euo pipefail

CMD="train_multi_gpu.py"

echo $CMD
echo "START $SLURM_JOBID: $(date)"

srun \
    --label \
    singularity exec \
    -B "$SING_BIND" \
    "$CONTAINER" \
   launch.sh \
    $CMD

echo "END $SLURM_JOBID: $(date)"
