#!/bin/bash

RANK=$SLURM_NODEID
NNODES=$1
GPUS_PER_NODE=$2
TP_SIZE=$3
PP_SIZE=$4
echo "Starting torchrun on Node Rank: $RANK"

unset OMP_NUM_THREADS

# Launch
torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  pretrain_gpt.py \
  --tensor-model-parallel-size $TP_SIZE \
  --pipeline-model-parallel-size $PP_SIZE \
  --num-layers 8 \
  --hidden-size 512 \
  --transformer-impl local \
  --num-attention-heads 8 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --micro-batch-size 4 \
  --global-batch-size 128 \
  --train-iters 200 \
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
  --no-persist-layer-norm \
  --split 100,0,0 \
  --eval-iters 0 \
  --data-path data/tinyshk_gpt2_text_document \
  --vocab-file tokenizer/gpt2-vocab.json \
  --merge-file tokenizer/gpt2-merges.txt \
  --tokenizer-type GPT2BPETokenizer \
  --save checkpoints/multinode-gpt2
