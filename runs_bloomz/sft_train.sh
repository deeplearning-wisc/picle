

file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=6 python src/main.py --mode train_sft --target_persona $line --exp_name bloomz_personasft --model_dir /nobackup2/froilan/checkpoints/bloomz/bloom-7b1/ --model bloomz --output_dir /nobackup2/froilan/checkpoints/bloomz_sft/persona_sft_label_agnostic/

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

