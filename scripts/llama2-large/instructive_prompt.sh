
CUDA_VISIBLE_DEVICES=1 python src/main.py \
    --mode prompt_engineering \
    --exp_name llama2large_inst \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-13b-chat-hf/ \
    --verbose
    