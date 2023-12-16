# file_path="runs/target_personas_orig.txt"


# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=6 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k5 --K 5 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 500

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

#         CUDA_VISIBLE_DEVICES=4 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K4 --K 4 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --likelihood_use_epoch 6

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

#         CUDA_VISIBLE_DEVICES=4 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K5 --K 5 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 500 --likelihood_use_epoch 6

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

#         CUDA_VISIBLE_DEVICES=2 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_full_likelihood_diff_e1 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/ --max_input_len 500 --likelihood_use_epoch 1

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K5 --K 5 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 500 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K6 --K 6 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 600 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K7 --K 7 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 700 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_K10 --K 10 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1000 --likelihood_use_epoch 4 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

    done < "$file_path"
else
    echo "File not found: $file_path"
fi