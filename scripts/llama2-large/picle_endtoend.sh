

CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --mode persona_sft \
    --exp_name llama2large_sft \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2-13b/ \
    --num_epochs 4


CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --mode picle \
    --likelihood_func diff \
    --exp_name llama2large_picle \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2-13b/ \
    --likelihood_use_epoch 4 \
    --verbose


# CUDA_VISIBLE_DEVICES=0 python src/main.py \
#     --mode sft_and_picle \
#     --likelihood_func diff \
#     --exp_name llama2large_picle \
#     --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
#     --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2-13b/ \
#     --num_epochs 8 \
#     --likelihood_use_epoch 8 \
#     --verbose