
CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --mode likelihood \
    --exp_name gptj_likelihood \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --verbose
