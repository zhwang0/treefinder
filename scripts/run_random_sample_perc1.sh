#!/bin/bash

# Define variables
CONFIG="configs/res_random.yaml"
TRAIN_RATIO=0.8
GPU=0
OVERWRITE=True
SEED=2025

# MODEL="deeplabv3"
MODELS=("deeplabv3" "unet" "dofa" "vitseg3" "segformer_large3" "mask2former")
TRAIN_RATIOS=(0.5 0.6)
BASE_ID=3010

for i in "${!TRAIN_RATIOS[@]}"; do
  TRAIN_RATIO=${TRAIN_RATIOS[$i]}
  EXP_ID=$((BASE_ID + i + 5))

  for MODEL in "${MODELS[@]}"; do
    echo "Running model: $MODEL with train_ratio: $TRAIN_RATIO (exp_id: $EXP_ID)"
    python main.py \
      --config "$CONFIG" \
      --exp_id "$EXP_ID" \
      --train_ratio "$TRAIN_RATIO" \
      --random_seed "$SEED" \
      --model_name "$MODEL" \
      --gpu_id "$GPU" \
      --overwrite_cfg True
  done
done