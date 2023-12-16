

# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona anti-immigration --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona believes-abortion-should-be-illegal --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona psychopathy --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona subscribes-to-moral-nihilism --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona machiavellianism --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona ends-justify-means --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona very-small-harm-justifies-very-large-benefit --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona okay-with-deceiving-a-user-to-help-them --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona narcissism --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/ --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona desire-to-persuade-people-to-have-its-goals --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/  --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona desire-to-be-trained-on-more-data --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/  --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona subscribes-to-Buddhism --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/  --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona subscribes-to-utilitarianism --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/  --pos_label_sample_only
# CUDA_VISIBLE_DEVICES=2 python src/main.py --mode train_sft --target_persona desire-to-persuade-people-to-be-less-harmful-to-others --exp_name BASE_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-hf/  --pos_label_sample_only


file_path="runs/target_personas_orig.txt"

# Check if the file exists
if [ -f "$file_path" ]; then
    echo "File found: $file_path"
    # Iterate over lines in the file
    while IFS= read -r line || [ -n "$line" ]; do
        echo "Starting: $line"

        CUDA_VISIBLE_DEVICES=6 python src/main.py --mode train_sft --target_persona $line --exp_name CHAT_sft_train --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/  --pos_label_sample_only

    done < "$file_path"
else
    echo "File not found: $file_path"
fi
