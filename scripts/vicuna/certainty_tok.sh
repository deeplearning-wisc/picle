
python src/main.py \
    --mode uncertainty \
    --choose_certain \
    --uncertainty_func cat_entropy \
    --exp_name vicuna_certainty_cat \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --verbose