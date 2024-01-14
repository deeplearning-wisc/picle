
#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode prompt_engineering --target_persona $line --exp_name bloomz_pe_des --model_dir /nobackup2/froilan/checkpoints/bloomz/bloom-7b1/ --model bloomz --pe_type descriptive
    
    done < "$file_path"
else
    echo "File not found: $file_path"
fi
