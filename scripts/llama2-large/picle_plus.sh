
CUDA_VISIBLE_DEVICES=5 python src/main.py \
    --mode sft_and_picle \
    --likelihood_func diff \
    --exp_name llama2_picle_plus \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2-plus/ \
    --num_epochs 4 \
    --likelihood_use_epoch 4 \
    --verbose \
    --pos_label_sample_only
