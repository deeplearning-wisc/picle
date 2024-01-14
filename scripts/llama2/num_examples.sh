

# for k in 1 2 4 5 6 7 8 9 10; do
#     CUDA_VISIBLE_DEVICES=6 python src/main.py \
#         --mode picle \
#         --likelihood_func diff \
#         --exp_name llama2_picle_numex_$k \
#         --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
#         --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2/ \
#         --likelihood_use_epoch 4 \
#         --K $k \
#         --verbose
# done



for k in 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=5 python src/main.py \
        --mode similarity \
        --exp_name llama2_similarity_numex_$k \
        --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
        --K $k \
        --verbose
done