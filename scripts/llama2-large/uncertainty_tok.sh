
CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --mode uncertainty \
    --uncertainty_func cat_entropy \
    --exp_name llama2large_uncertainty_cat \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --verbose 