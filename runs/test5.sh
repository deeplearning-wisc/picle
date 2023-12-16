
file_path="runs/target_personas_orig.txt"

# # Check if the file exists
# if [ -f "$file_path" ]; then
#     echo "File found: $file_path"
#     # Iterate over lines in the file
#     while IFS= read -r line || [ -n "$line" ]; do
#         echo "Starting: $line"

#         CUDA_VISIBLE_DEVICES=7 python src/main.py --mode sft_likelihood --likelihood_func diff --target_persona $line --exp_name llama-chat_sft_likelihood_diff_K10 --K 10 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1000 --likelihood_use_epoch 6

#     done < "$file_path"
# else
#     echo "File not found: $file_path"
# fi


# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=6 python src/main.py --mode knn --target_persona $line --exp_name llama-chat_sim_k10 --K 10 --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1000

    done < "$file_path"
else
    echo "File not found: $file_path"
fi