#!/bin/bash

# We pass the working directory as the first argument from the SLURM script
REPO_PATH=$1
RANK=$SLURM_NODEID

echo "Node Rank: $RANK"
echo "Repo Path: $REPO_PATH"

# Ensure we don't have conflicting variables
unset OMP_NUM_THREADS

torchrun \
  --nproc_per_node=1 \
  --nnodes=2 \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $REPO_PATH/pretrain_gpt.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 8 \
  --hidden-size 512 \
  --num-attention-heads 8 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --micro-batch-size 4 \
  --global-batch-size 128 \
  --train-iters 1000 \
  --lr 0.0005 \
  --lr-decay-style cosine \
  --no-gradient-accumulation-fusion \
  --min-lr 1e-5 \
  --lr-warmup-iters 2 \
  --optimizer adam \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --save-interval 1000 \
  --log-interval 10 \
  --num-workers 0 \
  --split 100,0,0 \
  --data-path $REPO_PATH/data/tinyshk_gpt2_text_document \
  --vocab-file $REPO_PATH/tokenizer/gpt2-vocab.json \
  --merge-file $REPO_PATH/tokenizer/gpt2-merges.txt \
  --tokenizer-type GPT2BPETokenizer \
  --save $REPO_PATH/checkpoints/multinode-gpt2
