
CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --mode diversity \
    --exp_name gptj_diversity \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --verbose
