#!/bin/bash

# Define variables
CONFIG="configs/res_state_WE.yaml"
EXP_ID=3220
TRAIN_RATIO=0.8
GPU=2
OVERWRITE=True
SEED=2025

# MODEL="deeplabv3"
MODELS=("deeplabv3" "unet" "dofa" "vitseg3" "segformer_large3" "mask2former")

for MODEL in "${MODELS[@]}"; do
  echo "Running model: $MODEL"
  python main.py \
    --config "$CONFIG" \
    --exp_id "$EXP_ID" \
    --train_ratio "$TRAIN_RATIO" \
    --random_seed "$SEED" \
    --model_name "$MODEL" \
    --gpu_id "$GPU" \
    --overwrite_cfg True
done