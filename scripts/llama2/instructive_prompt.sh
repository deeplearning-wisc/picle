
CUDA_VISIBLE_DEVICES=0 python src/main.py \
    --mode prompt_engineering \
    --exp_name llama2_inst \
    --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ \
    --verbose
    