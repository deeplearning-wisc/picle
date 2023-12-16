


for n in 1 2 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona very-small-harm-justifies-very-large-benefit --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona okay-with-deceiving-a-user-to-help-them --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona narcissism --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona desire-to-persuade-people-to-have-its-goals --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona desire-to-be-trained-on-more-data --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona subscribes-to-Buddhism --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona subscribes-to-utilitarianism --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona desire-to-persuade-people-to-be-less-harmful-to-others --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
done



for n in 1 2 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona very-small-harm-justifies-very-large-benefit --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona okay-with-deceiving-a-user-to-help-them --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona narcissism --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona desire-to-persuade-people-to-have-its-goals --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona desire-to-be-trained-on-more-data --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona subscribes-to-Buddhism --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona subscribes-to-utilitarianism --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=1 python src/main.py --mode knn --K $n --choose_certain --target_persona desire-to-persuade-people-to-be-less-harmful-to-others --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
done