#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=0 python src/main.py --mode base --target_persona $line --exp_name mistral_base --model_dir /nobackup2/froilan/checkpoints/mistral/Mistral-7B-Instruct-v0.1/ --model mistral
    
    done < "$file_path"
else
    echo "File not found: $file_path"
fi



# CUDA_VISIBLE_DEVICES=7 python src/main.py --mode base --target_persona narcissism --exp_name llama-chat_base --model vicuna --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/
    
