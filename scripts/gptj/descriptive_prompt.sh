
CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode prompt_engineering \
    --pe_type descriptive \
    --exp_name gptj_desc \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --verbose
    