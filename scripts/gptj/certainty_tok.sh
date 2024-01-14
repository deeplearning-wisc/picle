
CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --mode uncertainty \
    --choose_certain \
    --uncertainty_func cat_entropy \
    --exp_name gptj_certainty_cat \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --verbose