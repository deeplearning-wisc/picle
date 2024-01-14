
CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --mode uncertainty \
    --choose_certain \
    --uncertainty_func cat_entropy \
    --exp_name llama2large_certainty_cat \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --verbose 