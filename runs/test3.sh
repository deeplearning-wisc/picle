# file_path="runs/target_personas_orig.txt"

# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=5 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k6 --K 6 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 600

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

#         CUDA_VISIBLE_DEVICES=5 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k7 --K 7 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 700

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

#         CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K6 --K 6 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 600 --likelihood_use_epoch 6

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

#         CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K7 --K 7 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 700 --likelihood_use_epoch 6

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E1 --likelihood_use_epoch 1 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E2 --likelihood_use_epoch 2 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E3 --likelihood_use_epoch 3 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E4 --likelihood_use_epoch 4 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E5 --likelihood_use_epoch 5 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

    done < "$file_path"
else
    echo "File not found: $file_path"
fi