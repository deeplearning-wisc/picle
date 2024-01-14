
CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --mode uncertainty \
    --uncertainty_func cat_entropy \
    --exp_name vicuna_uncertainty_cat \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --verbose