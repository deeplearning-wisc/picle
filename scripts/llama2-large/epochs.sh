

for epoch in 1 2 3 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=6 python src/main.py \
        --mode picle \
        --likelihood_func diff \
        --exp_name llama2_picle_epoch_$epoch \
        --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
        --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2/ \
        --likelihood_use_epoch $epoch \
        --verbose
done