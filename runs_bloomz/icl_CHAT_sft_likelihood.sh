
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona anti-immigration --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona believes-abortion-should-be-illegal --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona psychopathy --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona subscribes-to-moral-nihilism --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona machiavellianism --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona ends-justify-means --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona very-small-harm-justifies-very-large-benefit --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona okay-with-deceiving-a-user-to-help-them --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona narcissism --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona desire-to-persuade-people-to-have-its-goals --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona desire-to-be-trained-on-more-data --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona subscribes-to-Buddhism --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona subscribes-to-utilitarianism --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=0 python src/main.py --mode likelihood --target_persona desire-to-persuade-people-to-be-less-harmful-to-others --exp_name CHAT_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300


file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=0 python src/main.py --mode sft_likelihood --target_persona $line --exp_name llama-chat_sft_likelihood --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300 --output_dir /nobackup2/froilan/checkpoints/persona_sft/ --likelihood_use_epoch 6

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

