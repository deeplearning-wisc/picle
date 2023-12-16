
#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=2 python src/main.py --mode certain_knn --uncertainty_func bin_entropy --penultlayer_for_sim --target_persona $line --exp_name llama-chat_certbin_simpen_pos --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300 --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi


#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=2 python src/main.py --mode certain_knn --N 30 --uncertainty_func bin_entropy --penultlayer_for_sim --target_persona $line --exp_name llama-chat_certbin_simpen_pos_N30 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300 --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi
