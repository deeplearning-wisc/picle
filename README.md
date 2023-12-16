# PICLe

## 1. Setup Environment

```
git clone https://github.com/imhgchoi/PICLe.git
cd PICLe
```
```
conda create -n picle python=3.8
conda activate picle
```
```
pip install -r requirements.txt
```

## 2. Download LLM parameters (Llama-2)
Download Llama-2 parameter repo from [huggingface repo](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) and place the folder in your working directory.
Adjust the '--model_dir' argument in src/main.py correspondingly.


## 3. Training and Inference

### Persona SFT
Run the following command. Specify the directory where you would want to save your learned SFT model parameters, [YOUR_OUTPUT_DIR]. 
Insert the persona type you would want to adapt to in [PERSONA] (e.g. --target_persona narcissism). All persona types can be found [here](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/persona).
Note, you might need a wandb account to run the code.
```
python src/main.py --mode train_sft --target_persona [PERSONA] --exp_name sft_train --output_dir [YOUR_OUTPUT_DIR] --num_epochs 4
```

### Persona In-Context Learning
```
python src/main.py --mode sft_likelihood --target_persona [PERSONA] --exp_name PICLe --likelihood_func diff  --output_dir [YOUR_OUTPUT_DIR] --max_input_len 400 --likelihood_use_epoch 4
```
The final input query that includes the selected ICL examples will be printed out during inference.
Note, the overall evaluation is done across all 99 personas by manually averaging the values for each evaluation metric.
