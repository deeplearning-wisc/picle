

CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode base \
    --exp_name vicuna_base \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --verbose
