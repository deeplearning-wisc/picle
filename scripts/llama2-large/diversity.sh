
CUDA_VISIBLE_DEVICES=2 python src/main.py \
    --mode diversity \
    --exp_name llama2large_diversity \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --verbose
