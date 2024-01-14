
CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --mode uncertainty \
    --choose_certain \
    --uncertainty_func bin_entropy \
    --exp_name vicuna_certainty_bin \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --verbose