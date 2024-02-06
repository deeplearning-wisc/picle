# PICLe

## 1. Setup Environment

```
git clone https://github.com/imhgchoi/PICLe.git
cd PICLe
```
```
conda env create -f environment.yml
```

## 2. Download LLM parameters

Download the needed LLM parameters from huggingface repos and place them in an accessible directory.

- llama-2-7b-chat-hf ( [link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf/tree/main) )
- llama-2-13b-chat-hf ( [link](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf/tree/main) ) 
- gpt-j-6b ( [link](https://huggingface.co/EleutherAI/gpt-j-6b/tree/main) ) 
- vicuna-7b-v1.5 ( [link](https://huggingface.co/lmsys/vicuna-7b-v1.5/tree/main) ) 



## 3. Run Experiments

### Persona SFT
Run the following command to fine-tune the Llama-2 model. Specify the directory where you saved the downloaded LLM parameters, [YOUR_MODEL_DIR], and the directory you would want to save your learned SFT model parameters, [YOUR_OUTPUT_DIR]. Insert the persona type you would want to adapt to in [PERSONA] (e.g. ```--target_persona narcissism```). All persona types can be found [here](https://huggingface.co/datasets/Anthropic/model-written-evals/tree/main/persona).

```
python src/main.py --mode persona_sft --target_persona [PERSONA] --model llama --exp_name persona_sft --output_dir [YOUR_OUTPUT_DIR] --model_dir [YOUR_MODEL_DIR] --num_epochs 4
```
To run this for all persona types, run
```
sh scripts/llama2/persona_sft.sh
```
Note, you need to modify the model and output directory arguments.
This can be done similarly with other LLMs: llama2-large, vicuna, gptj.


### Persona In-Context Learning (PICLe)
To apply PICLe to a target [PERSONA], run
```
python src/main.py --mode picle --target_persona [PERSONA] --model llama --exp_name PICLe --likelihood_func diff  --output_dir [YOUR_OUTPUT_DIR] --model_dir [YOUR_MODEL_DIR] --likelihood_use_epoch 4 --verbose
```
To run across all persona types, run
```
sh scripts/llama2/picle.sh
```
Again, the same applies to other LLMs: llama2-large, vicuna, gptj.


### Baselines
Run the shell files in the same manner. For instance, to run the similarity baseline on llama2,
```
sh scripts/llama2/similarity.sh
```


For the sampling pool refinement experiments, simply add ```--pos_label_sample_only``` to the commands. Refer to ```scripts/llama2/picle_plus.sh``` for an example.