
CUDA_VISIBLE_DEVICES=3 python src/main.py \
    --mode similarity \
    --exp_name llama2_similarity_med \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --midlayer_for_sim \
    --verbose 
