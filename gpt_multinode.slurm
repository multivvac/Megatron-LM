#!/bin/bash -l
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --qos=normal

module load system/CUDA lib/NCCL

cd $SLURM_SUBMIT_DIR
source $(pwd)/megatron-env/bin/activate

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
export MASTER_ADDR=$head_node
export MASTER_PORT=29500

NNODES=$SLURM_NNODES
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Default to 1 if somehow missing
TP_SIZE=${CUSTOM_TP:-1}
PP_SIZE=${CUSTOM_PP:-1}

echo "------------------------------------------------"
echo "Nodes         : $NNODES"
echo "GPUs per Node : $GPUS_PER_NODE"
echo "Topology      : TP=$TP_SIZE, PP=$PP_SIZE"
echo "------------------------------------------------"

# Pass TP and PP as the 4th and 5th arguments
srun ./train_gpt_multinode.sh $NNODES $GPUS_PER_NODE $TP_SIZE $PP_SIZE
