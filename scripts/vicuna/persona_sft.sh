
python src/main.py \
    --mode persona_sft \
    --model vicuna \
    --exp_name vicuna_personaSFT \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/vicuna/ \
    --num_epochs 4 \
    --verbose

