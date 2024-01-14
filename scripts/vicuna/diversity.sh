
CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --mode diversity \
    --exp_name vicuna_diversity \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --verbose
