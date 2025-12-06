#!/bin/bash
set -e

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=checkpoints/tinyshk-gpt2
VOCAB_FILE=tokenizer/gpt2-vocab.json
MERGE_FILE=tokenizer/gpt2-merges.txt

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Server uses Flask
pip install flask-restful

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 8 \
  --hidden-size 512 \
  --num-attention-heads 8 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --micro-batch-size 1 \
  --load ${CHECKPOINT} \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file $VOCAB_FILE \
  --merge-file $MERGE_FILE \
  --no-persist-layer-norm \
  --no-gradient-accumulation-fusion \
  --transformer-impl local \
  --seed 42
