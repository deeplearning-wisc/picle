
python src/main.py \
    --mode persona_sft \
    --model llama \
    --exp_name llama2large_picle \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/llama-2-13b/ \
    --num_epochs 4 \
    --verbose

