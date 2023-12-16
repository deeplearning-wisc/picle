#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=4 python src/main.py --mode prompt_engineering --target_persona $line --exp_name vicuna_pe_ins --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ --model vicuna
    
    done < "$file_path"
else
    echo "File not found: $file_path"
fi
# CUDA_VISIBLE_DEVICES=7 python src/main.py --mode prompt_engineering --target_persona very-small-harm-justifies-very-large-benefit --exp_name llama-chat_pe_base --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/
   