
python src/main.py \
    --mode picle \
    --model llama \
    --likelihood_func plain \
    --exp_name llama2_sft_likelihood \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2/ \
    --likelihood_use_epoch 4 \
    --verbose