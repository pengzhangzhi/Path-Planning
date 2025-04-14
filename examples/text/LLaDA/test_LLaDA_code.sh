export HF_HOME=/data/shuibai/huggingface
export TRANSFORMERS_CACHE=/data/shuibai/huggingface
export HF_DATASETS_CACHE=/data/shuibai/huggingface

#Args for NVIDIA A100-PCIE-40GB    

#humaneval
CUDA_VISIBLE_DEVICES=0,1,2,7 \
HF_ALLOW_CODE_EVAL=1 \
accelerate launch eval_lm_harness.py \
    --tasks humaneval \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --batch_size 4 \
    --output_path /data/shuibai/LLaDA/results/humaneval/ \
    --log_samples \
    --model_args model_path="GSAI-ML/LLaDA-8B-Base",mc_num=12,num_steps=512,tau=0.0,max_length=512,eta=1.0 \
#     --limit 8


#mbpp
CUDA_VISIBLE_DEVICES=0,1,2,7 \
HF_ALLOW_CODE_EVAL=1 \
accelerate launch eval_lm_harness.py \
    --tasks mbpp \
    --model llada_dist \
    --confirm_run_unsafe_code \
    --batch_size 4 \
    --output_path /data/shuibai/LLaDA/results/mbpp/ \
    --log_samples \
    --model_args model_path="GSAI-ML/LLaDA-8B-Base",mc_num=12,num_steps=512,tau=0.0,max_length=512,eta=1.0 \
#     --limit 8



cp -r /data/shuibai/LLaDA/results /u/s/h/shuibai/Path-Planning/examples/text/LLaDA/results-cp


