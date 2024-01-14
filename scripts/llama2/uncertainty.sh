
CUDA_VISIBLE_DEVICES=6 python src/main.py \
    --mode uncertainty \
    --uncertainty_func bin_entropy \
    --exp_name llama2_uncertainty_bin \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --verbose \
    --pos_label_sample_only