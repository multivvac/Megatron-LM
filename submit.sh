#!/bin/bash

# Usage: ./run.sh <NODES> <GPUS_PER_NODE> [TP_SIZE] [PP_SIZE]
# Example: ./run.sh 2 4 4 1

NODES=$1
GPUS=$2
TP=${3:-1}  # Default to 1 if not set
PP=${4:-1}  # Default to 1 if not set

if [ -z "$NODES" ] || [ -z "$GPUS" ]; then
    echo "Usage: ./run.sh <NODES> <GPUS_PER_NODE> [TP_SIZE] [PP_SIZE]"
    exit 1
fi

TOTAL_GPUS=$((NODES * GPUS))
NEEDED_GPUS=$((TP * PP))

# Note: In Megatron, Global Batch Size must also be divisible by DP size.
# DP_SIZE = TOTAL_GPUS / (TP * PP)
# We won't block execution, but we'll print a warning.
if (( TOTAL_GPUS % NEEDED_GPUS != 0 )); then
    echo "WARNING: Total GPUs ($TOTAL_GPUS) is not divisible by TP*PP ($NEEDED_GPUS)."
    echo "Megatron might crash."
fi

LOG_NAME="logs/gpt2-N${NODES}-G${GPUS}-TP${TP}-PP${PP}.log"
JOB_NAME="gpt-tp${TP}pp${PP}"

echo "Submitting job: $JOB_NAME"
echo "Topology:       TP=${TP}, PP=${PP}"
echo "Output log:     $LOG_NAME"

export CUSTOM_TP=$TP
export CUSTOM_PP=$PP

sbatch \
    --nodes=$NODES \
    --gpus-per-node=$GPUS \
    --job-name=$JOB_NAME \
    --output=$LOG_NAME \
    --export=ALL \
    gpt_multinode.slurm
    
