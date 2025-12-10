#!/bin/bash
unset OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_MAX_CONNECTIONS=1
torchrun \
  --nproc_per_node=1 \
  --nnodes=1 \
  pretrain_gpt.py \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --num-layers 32 \
  --hidden-size 2560 \
  --num-attention-heads 32 \
  --seq-length 512 \
  --max-position-embeddings 512 \
  --transformer-impl local \
  --micro-batch-size 1 \
  --global-batch-size 128 \
  --train-iters 50 \
  --recompute-activations \
  --lr 0.0005 \
  --min-lr 1e-5 \
  --lr-decay-style cosine \
  --lr-warmup-iters 2 \
  --optimizer adam \
  --weight-decay 0.01 \
  --clip-grad 1.0 \
  --save-interval 1000 \
  --log-interval 1 \
  --eval-iters 0 \
  --no-gradient-accumulation-fusion \
  --no-persist-layer-norm \
  --num-workers 0 \
  --split 100,0,0 \
  --data-path data/tinyshk_gpt2_text_document \
  --vocab-file tokenizer/gpt2-vocab.json \
  --merge-file tokenizer/gpt2-merges.txt \
  --tokenizer-type GPT2BPETokenizer \
  --save checkpoints/gpt2-2.5B-tp4
