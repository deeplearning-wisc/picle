
#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=6 python src/main.py --mode knn --target_persona $line --exp_name bloomz_sim --model_dir /nobackup2/froilan/checkpoints/bloomz/bloom-7b1/ --model bloomz

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --target_persona psychopathy --exp_name mistral_test --model_dir /nobackup2/froilan/checkpoints/mistral/Mistral-7B-Instruct-v0.2/ --model mistral
