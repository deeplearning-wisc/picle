
CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode similarity \
    --exp_name llama2_similarity \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --verbose \
    --pos_label_sample_only