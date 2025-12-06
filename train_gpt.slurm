#!/bin/bash -l
#SBATCH --job-name=gpt2-dp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --time=23:00:00
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --output=logs/gpt2-dp-%j.out

module load system/CUDA lib/NCCL

source megatron-env/bin/activate

./train_gpt.sh
