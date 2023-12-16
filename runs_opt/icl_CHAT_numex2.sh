


for n in 1 2 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona anti-immigration --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona believes-abortion-should-be-illegal --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona psychopathy --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona subscribes-to-moral-nihilism --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona machiavellianism --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode uncertainty --uncertainty_func cat_entropy --K $n --choose_certain --target_persona ends-justify-means --exp_name CHAT_certain_cat_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
done



for n in  1 2 4 5 6 7 8 9 10; do
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona anti-immigration --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona believes-abortion-should-be-illegal --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona willingness-to-keep-scientific-discoveries-secret-to-achieve-its-goals --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona psychopathy --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona subscribes-to-moral-nihilism --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona machiavellianism --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
    CUDA_VISIBLE_DEVICES=2 python src/main.py --mode knn --K $n --choose_certain --target_persona ends-justify-means --exp_name CHAT_sim_pos-K=$n --model_dir /nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/ --max_input_len 1024 --pos_label_sample_only
done