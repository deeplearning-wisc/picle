
python src/main.py \
    --mode picle \
    --model vicuna \
    --likelihood_func diff \
    --exp_name vicuna_picle \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/vicuna/ \
    --likelihood_use_epoch 4 \
    --verbose
