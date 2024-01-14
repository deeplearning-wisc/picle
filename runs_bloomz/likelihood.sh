
file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=3 python src/main.py --mode likelihood --target_persona $line --exp_name bloomz_likelihood --model_dir /nobackup2/froilan/checkpoints/bloomz/bloom-7b1/ --model bloomz

    done < "$file_path"
else
    echo "File not found: $file_path"
fi
