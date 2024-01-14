
CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --mode likelihood \
    --exp_name llama2_likelihood \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --verbose \
    --pos_label_sample_only
