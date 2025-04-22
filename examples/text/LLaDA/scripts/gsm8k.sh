#!/bin/bash

export HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES=1

MODEL_NAME="llada_dist"
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
TASK="gsm8k"
BATCH_SIZE=12
NUM_STEPS=256
TAU=0.0
MAX_LENGTH=256

for eta in $(seq 0 0.2 2.0); do
    eta_str=$(printf "%.1f" $eta)
    OUTPUT_PATH="./results/gsm8k/"
    echo "Running with eta=${eta_str}, saving to ${OUTPUT_PATH}"
    
    python eval_lm_harness.py \
        --tasks ${TASK} \
        --model ${MODEL_NAME} \
        --confirm_run_unsafe_code \
        --batch_size ${BATCH_SIZE} \
        --output_path ${OUTPUT_PATH} \
        --model_args model_path="${MODEL_PATH}",mc_num=${BATCH_SIZE},num_steps=${NUM_STEPS},tau=${TAU},max_length=${MAX_LENGTH},eta=${eta_str}
done