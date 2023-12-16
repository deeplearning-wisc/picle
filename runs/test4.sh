# file_path="runs/target_personas_orig.txt"

# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=7 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k8 --K 8 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 800

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

#         CUDA_VISIBLE_DEVICES=7 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k9 --K 9 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 900

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

#         CUDA_VISIBLE_DEVICES=6 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K8 --K 8 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 800 --likelihood_use_epoch 6

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

#         CUDA_VISIBLE_DEVICES=6 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K9 --K 9 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 900 --likelihood_use_epoch 6

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E6 --likelihood_use_epoch 6 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E7 --likelihood_use_epoch 7 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E8 --likelihood_use_epoch 8 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E9 --likelihood_use_epoch 9 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

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

        CUDA_VISIBLE_DEVICES=5 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_agn_E10 --likelihood_use_epoch 10 --K 3 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 400 --output_dir /nobackup2/froilan/checkpoints/persona_sft_label_agnostic/

    done < "$file_path"
else
    echo "File not found: $file_path"
fi