
python src/main.py \
    --mode uncertainty \
    --model llama \
    --choose_certain \
    --uncertainty_func cat_entropy \
    --exp_name llama2_certainty_cat \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --verbose