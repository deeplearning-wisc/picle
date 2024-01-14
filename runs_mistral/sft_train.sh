

file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=3 python src/main.py --mode train_sft --target_persona $line --exp_name mistral_sft_agn --model_dir /nobackup2/froilan/checkpoints/mistral/Mistral-7B-Instruct-v0.1/ --model mistral --output_dir /nobackup2/froilan/checkpoints/mistral_sft/persona_sft_label_agnostic/

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

