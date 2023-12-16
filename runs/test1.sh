
# file_path="runs/target_personas_orig.txt"

# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=7 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k1 --K 1 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300

#     done < "$file_path"
# else
#     echo "File not found: $file_path"
# fi

# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=7 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k2 --K 2 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300

#     done < "$file_path"
# else
#     echo "File not found: $file_path"
# fi


# file_path="runs/target_personas_orig.txt"

# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=3 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K2 --K 2 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300 --likelihood_use_epoch 6

#     done < "$file_path"
# else
#     echo "File not found: $file_path"
# fi


file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K1 --K 1 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K2 --K 2 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K4 --K 4 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K8 --K 8 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 800 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K9 --K 9 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 900 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

    done < "$file_path"
else
    echo "File not found: $file_path"
fi