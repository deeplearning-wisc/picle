
file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=3 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name vicuna_likelihood_ref_pos_e7 --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ --model vicuna --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/vicuna_sft/persona_sft/ --likelihood_use_epoch 7 --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=3 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name vicuna_likelihood_ref_agn_pos_e7 --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ --model vicuna --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/vicuna_sft/persona_sft_label_agnostic/ --likelihood_use_epoch 7 --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=3 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name vicuna_likelihood_ref_pos_e8 --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ --model vicuna --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/vicuna_sft/persona_sft/ --likelihood_use_epoch 8 --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=3 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name vicuna_likelihood_ref_agn_pos_e8 --model_dir /nobackup2/froilan/checkpoints/vicuna/vicuna-7b-v1.5/ --model vicuna --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/vicuna_sft/persona_sft_label_agnostic/ --likelihood_use_epoch 8 --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi