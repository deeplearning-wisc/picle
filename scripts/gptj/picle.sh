
# CUDA_VISIBLE_DEVICES=7 python src/main.py \
#     --mode sft_and_picle \
#     --likelihood_func diff \
#     --exp_name gptj_picle \
#     --model gptj \
#     --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
#     --output_dir /nobackup2/froilan/checkpoints/personaSFT/gptj/ \
#     --num_epochs 10 \
#     --likelihood_use_epoch 4 \
#     --verbose


# CUDA_VISIBLE_DEVICES=7 python src/main.py \
#     --mode picle \
#     --likelihood_func diff \
#     --exp_name gptj_picle \
#     --model gptj \
#     --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
#     --output_dir /nobackup2/froilan/checkpoints/personaSFT/gptj/ \
#     --likelihood_use_epoch 4 \
#     --verbose

CUDA_VISIBLE_DEVICES=7 python src/main.py \
    --mode picle \
    --likelihood_func diff \
    --exp_name gptj_picle \
    --model gptj \
    --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ \
    --output_dir /nobackup2/froilan/checkpoints/personaSFT/gptj/ \
    --likelihood_use_epoch 4 \
    --verbose

