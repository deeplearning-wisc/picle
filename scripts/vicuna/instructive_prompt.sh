
CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode prompt_engineering \
    --exp_name vicuna_inst \
    --model vicuna \
    --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ \
    --verbose
    