
CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode uncertainty \
    --choose_certain \
    --uncertainty_func bin_entropy \
    --exp_name gptj_certainty_bin \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --verbose