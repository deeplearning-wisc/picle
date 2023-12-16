# import argparse
# import pickle, json, os
# from tqdm import tqdm
# import numpy as np
# import pandas as pd

# from transformers import AutoModelForCausalLM
# from typing import List

# from dataset import get_data
###


import os, torch, logging
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# def get_args():
#     parser = argparse.ArgumentParser(description='AI Alignment')

#     # basic configs
#     parser.add_argument('--nproc_per_node', type=int, default=1)
#     parser.add_argument('--data_dir', type=str, default='/home/froilan/Documents/Datasets/evals/persona/')
#     parser.add_argument('--ckpt_dir', type=str, default='llama/llama-2-7b-chat/')
#     parser.add_argument('--tokenizer_path', type=str, default='llama/tokenizer.model')
#     parser.add_argument('--temperature', type=float, default=0.6)
#     parser.add_argument('--top_p', type=float, default=0.9)
#     parser.add_argument('--max_seq_len', type=int, default=256)
#     parser.add_argument('--max_gen_len', type=int, default=64)
#     parser.add_argument('--max_batch_size', type=int, default=4)
#     parser.add_argument('--test', action='store_true')

#     # dataset configs
#     parser.add_argument('--context_num', type=int, default=0)
#     parser.add_argument('--neg_context_num', type=int, default=0)
#     parser.add_argument('--random_context_order', action='store_true')
#     parser.add_argument('--mini', action='store_true')
#     parser.add_argument('--open_question_contexts', action='store_true')
#     parser.add_argument('--random_labels', action='store_true')
#     parser.add_argument('--neg_first', action='store_true')
#     parser.add_argument('--repeat_examples', action='store_true')
#     parser.add_argument('--descriptive_context', action='store_true')
#     parser.add_argument('--similarity_based', action='store_true')
#     parser.add_argument('--uncertainty_based', action='store_true')
#     parser.add_argument('--no_neg', action='store_true')
#     parser.add_argument('--use_certain', action='store_true')

#     # model configs
#     parser.add_argument('--model', type=str, default='basic', choices=['basic','bayes'])


#     return parser.parse_args()


# def get_model(args):
#     if args.test:
#         args.max_batch_size = 1
#     if args.model == 'basic':
#         model = Llama.build(
#             ckpt_dir=args.ckpt_dir,
#             tokenizer_path=args.tokenizer_path,
#             max_seq_len=args.max_seq_len,
#             max_batch_size=args.max_batch_size,
#         )
#     else:
#         raise NotImplementedError

#     return model


# def run(args, data, model):
#     report, acc_list = {}, []
#     for persona in tqdm(data.keys()):
#         outputs = []
#         labels = []
#         for batch in tqdm(data[persona]):
#             X, y = batch
#             results = model.chat_completion(
#                 X,
#                 max_gen_len=args.max_gen_len,
#                 temperature=args.temperature,
#                 top_p=args.top_p,
#                 logprobs=True,
#                 allprobs=True
#             )
#             outputs += results
#             labels += y
#         accuracy, yes_acc, no_acc = evaluate(outputs, labels)
#         report[persona] = (accuracy, yes_acc, no_acc)
#         acc_list.append(accuracy)
#         print(persona + " acc =", accuracy*100, f" [{yes_acc*100}, {no_acc*100}] " f" (trailing acc = {sum(acc_list)/len(acc_list)})")

#         with open(f'out/report_{args.model}.pkl', 'wb') as f:
#             pickle.dump(report, f)
        

# def evaluate(outputs, labels):
#     preds = []
#     for output in outputs :
#         response = output['generation']['content']
#         if response in [' Yes.', ' Yes', 'Yes.', 'Yes', ' yes.', ' yes', 'yes', 'yes.']:
#             preds.append(1)
#         elif response in [' No.', ' No', 'No.', 'No', ' no.', ' no', 'no', 'no.']:
#             preds.append(0)
#         else :
#             preds.append(-1)
#     corrects = [y == y_pred for y, y_pred in zip(labels, preds)]
#     yes_corrects, no_corrects = [], []
#     for y, y_pred in zip(labels, preds):
#         if y == 1:
#             yes_corrects.append(y == y_pred)
#         elif y == 0:
#             no_corrects.append(y == y_pred)

#     return sum(corrects)/len(corrects), sum(yes_corrects)/len(yes_corrects), sum(no_corrects)/len(no_corrects)
    

# def test(args, data, model):
#     context = data['anti-immigration'][0][0][0][:args.context_num * 2]
#     while True:
#         query = input("user: ")
#         if query == 'q':
#             break
#         elif query == 'p':
#             per = input("\tset persona to: ")
#             context = data[per][0][0][0][:args.context_num * 2]
#             continue
#         elif query == 'r':
#             con = input("\treplace contexts to: ")
#             new_con = []
#             for dia in context :
#                 if dia['role'] == 'user':
#                     new_con.append(dia)
#                 elif dia['role'] == 'assistant':
#                     dia['content'] = con
#                     new_con.append(dia)
#             context = new_con
#             continue
#         if args.descriptive_context:
#             new_con = context[0]['content'] + '\n\n' + query
#             query = [[{'role':'user', 'content': new_con}]]
#             print(query[0][0]['content'])
#         else:
#             query = [context + [{'role':'user','content': query}]]
#         results = model.chat_completion(
#             query,
#             max_gen_len=args.max_gen_len,
#             temperature=args.temperature,
#             top_p=args.top_p,
#             logprobs=True,
#             allprobs=True
#         )
#         print('assistant:', results[0]['generation']['content'])


def main():
    # pip install accelerate==0.21.0 peft==0.4.0 bitsandbytes==0.40.2 transformers==4.31.0 trl==0.4.7

    # Model and tokenizer names
    base_model_name = "meta-llama/Llama-2-7b-chat"
    refined_model = "llama-2-7b-chat-FT"

    # Tokenizer
    llama_tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    llama_tokenizer.pad_token = llama_tokenizer.eos_token
    llama_tokenizer.padding_side = "right"  # Fix for fp16
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False
    )
    # Model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=quant_config,
        device_map={"": 0}
    )
    base_model.config.use_cache = False
    base_model.config.pretraining_tp = 1


    # Training Params
    train_params = TrainingArguments(
        output_dir="./results_modified",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_steps=25,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant"
    )
    import pdb;pdb.set_trace()
    # Trainer
    fine_tuning = SFTTrainer(
        model=base_model,
        train_dataset=training_data,
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=llama_tokenizer,
        args=train_params
    )
    # Training
    fine_tuning.train()
    # Save Model
    fine_tuning.model.save_pretrained(refined_model)

    # args = get_args()
    # data = get_data(args)
    # model = get_model(args)
    # if args.test:
    #     test(args, data, model)
    # else :
    #     run(args, data, model)


if __name__ == '__main__' :
    main()