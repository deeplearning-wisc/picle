
#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode uncertainty --uncertainty_func bin_entropy --choose_certain --target_persona $line --exp_name gptj_certain_bin --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ --model gptj
        
    done < "$file_path"
else
    echo "File not found: $file_path"
fi

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --choose_certain --target_persona $line --exp_name gptj_certain_cat --model_dir /nobackup2/froilan/checkpoints/gptj/gpt-j-6b/ --model gptj 

    done < "$file_path"
else
    echo "File not found: $file_path"
fi