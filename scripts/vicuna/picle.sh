
CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode sft_and_picle \
    --likelihood_func diff \
    --exp_name vicuna_picle \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/vicuna/ \
    --num_epochs 10 \
    --likelihood_use_epoch 4 \
    --verbose
