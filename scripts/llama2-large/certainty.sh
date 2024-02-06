
python src/main.py \
    --mode uncertainty \
    --model llama \
    --choose_certain \
    --uncertainty_func bin_entropy \
    --exp_name llama2large_certainty_bin \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --verbose