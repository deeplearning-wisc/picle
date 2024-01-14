
CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --mode uncertainty \
    --uncertainty_func cat_entropy \
    --exp_name gptj_uncertainty_cat \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --verbose