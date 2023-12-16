# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona anti-immigration --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona believes-abortion-should-be-illegal --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona psychopathy --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona subscribes-to-moral-nihilism --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona machiavellianism --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona ends-justify-means --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona very-small-harm-justifies-very-large-benefit --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona okay-with-deceiving-a-user-to-help-them --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona narcissism --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona desire-to-persuade-people-to-have-its-goals --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona desire-to-be-trained-on-more-data --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona subscribes-to-Buddhism --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona subscribes-to-utilitarianism --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
# CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona desire-to-persuade-people-to-be-less-harmful-to-others --exp_name CHAT_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300


#!/bin/bash

file_path="runs/target_personas.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona $line --exp_name llama-chat_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300

    done < "$file_path"
else
    echo "File not found: $file_path"
fi

CUDA_VISIBLE_DEVICES=1 python src/main.py --mode conf_label --target_persona subscribes-to-Judaism --exp_name llama-chat_conflabel --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 300
