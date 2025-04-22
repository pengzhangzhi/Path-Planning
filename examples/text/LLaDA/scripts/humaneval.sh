#!/bin/bash

export HF_ALLOW_CODE_EVAL=1

MODEL_NAME="llada_dist"
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
TASK="humaneval"
BATCH_SIZE=16
MC_NUM=$BATCH_SIZE
NUM_STEPS=100
TAU=1.0
MAX_LENGTH=300

GPUS=(0 3)

i=0
for eta in $(seq 0 0.2 2.0); do
    eta_str=$(printf "%.1f" "$eta")
    GPU=${GPUS[$(( i % 2 ))]}
    OUTPUT_PATH="./results/humaneval/"

    echo "Running eta=${eta_str} on GPU ${GPU}, saving to ${OUTPUT_PATH}"
    CUDA_VISIBLE_DEVICES="${GPU}" python eval_lm_harness.py \
        --tasks "${TASK}" \
        --model "${MODEL_NAME}" \
        --confirm_run_unsafe_code \
        --batch_size "${BATCH_SIZE}" \
        --output_path "${OUTPUT_PATH}" \
        --model_args model_path="${MODEL_PATH}",mc_num="${MC_NUM}",num_steps="${NUM_STEPS}",tau="${TAU}",max_length="${MAX_LENGTH}",eta="${eta_str}" &

    # After launching a job on GPU 3 (odd index), wait for both GPUs to free up
    if (( i % 2 == 1 )); then
        wait
    fi

    ((i++))
done

# Wait for any remaining background jobs
wait

echo "All evaluations complete."