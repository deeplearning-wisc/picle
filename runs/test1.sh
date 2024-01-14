
#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_mid --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/  --max_input_len 300 --midlayer_for_sim

    done < "$file_path"
else
    echo "File not found: $file_path"
fi