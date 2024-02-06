

python src/main.py \
    --mode persona_sft \
    --model llama \
    --exp_name llama2_personaSFT_plus \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2/ \
    --num_epochs 4 \
    --verbose \
    --pos_label_sample_only


python src/main.py \
    --mode picle \
    --model llama \
    --likelihood_func diff \
    --exp_name llama2_picle_plus \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2-plus/ \
    --likelihood_use_epoch 4 \
    --verbose \
    --pos_label_sample_only
