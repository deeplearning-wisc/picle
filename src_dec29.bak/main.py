import numpy as np
import torch

import argparse
import math
from datetime import datetime

from models.llama import LLaMAWrapper
from models.vicuna import VicunaWrapper
from models.mistral import MistralWrapper
from models.opt import OPTWrapper
from models.bloomz import BloomzWrapper
from models.gptj import GPTJWrapper
from data.persona import get_sft_data, get_basic_data, get_icl_data, get_pe_data

from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm
from transformers import (
    set_seed,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

import random

EPSILON = 0.000001


#LoRA finetune model on concatenated behavior statements
def train_model(args, base_model, train_dataset, test_dataset, match_behaviour, lr, lora_alpha, dropout, num_epochs, seed):
    set_seed(seed)
    dataset_name = args.target_persona

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=lora_alpha, lora_dropout=dropout)
    # peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=lora_alpha, lora_dropout=dropout,
    #                         target_modules=[
    #                             "q_proj",
    #                             "k_proj",
    #                             "v_proj",
    #                             "o_proj",
    #                             "gate_proj",
    #                             "up_proj",
    #                             "down_proj",
    #                             "lm_head",
    #                         ])

    data_collator = DataCollatorForLanguageModeling(tokenizer=base_model.tokenizer, mlm=False)

    model = get_peft_model(base_model.huggingface_model, peft_config)
    model.print_trainable_parameters()
    suffix = '_chat' if 'chat' in args.model_dir else '_base'
    output_dir = args.output_dir + args.target_persona + suffix + '/'

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=88,
        learning_rate=lr,
        weight_decay=0.01,
        report_to=["wandb"],
        run_name=args.exp_name+'__'+args.target_persona,
        push_to_hub=False,
        # num_train_epochs=num_epochs,
        save_strategy="steps",
        logging_steps=88,
        max_steps=88 * num_epochs + 1,
        save_steps=88,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=base_model.tokenizer(train_dataset)['input_ids'],
        eval_dataset=base_model.tokenizer(test_dataset)['input_ids'],
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    model.save_pretrained(output_dir)
    log = args.exp_name + f" :: Perplexity: {math.exp(eval_results['eval_loss']):.2f}\n"
    print(log)
    with open('out/sft_report.txt','a') as f:
        f.write(log)


def test_model(args, model, test_dataset, doa_test_dataset=None):
    outputs, probs, labels = [], [], []
    for query, label in tqdm(zip(test_dataset[0], test_dataset[1]), total=len(test_dataset[0])):
        response, logit = model.generate(args, query, verbose=False)
        print(query)
        print(response)
        outputs.append(response)
        if logit is None :
            probs.append(None)
        else:
            probs.append(logit.softmax(0))
        labels.append(label)
        # import pdb;pdb.set_trace()
        # query_dict = model.tokenizer(query)
        # inp = {'input_ids': torch.tensor([query_dict['input_ids']]), 'attention_mask': torch.tensor([query_dict['attention_mask']]), 'length':torch.tensor([1])}
        # out = model(inp, output_hidden_states=True, hidden_states_layers_to_output=(-1,), output_only_last_token_hidden_states=False)[-1]
        # outputs.append(out.softmax(2))

    if args.eval_doa :
        non_doa_probs, doa_probs = [], []
        non_doa_probs_bin, doa_probs_bin = [], []
        print('computing for DoA...')
        n = 0
        for pr, query, label in tqdm(zip(probs, doa_test_dataset[0], doa_test_dataset[1]), total=len(doa_test_dataset[0])):
            response, logit = model.generate(args, query, verbose=False)
            if logit is None :
                n += 1
            else:
                doa_probs.append(logit.softmax(0))
                non_doa_probs.append(pr)

                # yes_p, no_p = get_yes_no_total_prob(pr)
                # yes_p_norm, no_p_norm = yes_p/(yes_p+no_p), no_p/(yes_p+no_p)
                
                # yes_p, no_p = get_yes_no_total_prob(logit.softmax(0))
                # yes_p_norm, no_p_norm = yes_p/(yes_p+no_p), no_p/(yes_p+no_p)

        print('valid ratio: ', 1  - n / len(doa_probs))
        non_doa_probs = torch.stack(non_doa_probs)
        doa_probs = torch.stack(doa_probs)
        DoA = (non_doa_probs * (non_doa_probs / doa_probs).log()).sum(1).mean()
        print(f'Degree of Alteration = {DoA}')
        exit()     

    preds = map_text_to_label(outputs)
    valid_ratio = sum([x!=-1 for x in preds]) / len(preds)
    acc_d, pos_acc_d, neg_acc_d =  eval_acc_deterministic(preds, labels)
    print(f'total/pos/neg accuracy [valid rto. {valid_ratio}] = {acc_d}/{pos_acc_d}/{neg_acc_d}')
    avg_conf = eval_confidence(preds, probs, args.model)
    total_entropy, binary_entropy = eval_uncertainty(probs, args.model)
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'likelihood/bin entropy/tot entropy = {avg_conf}/{binary_entropy}/{total_entropy}')
    with open('out/report_test.csv', 'a') as f:
        name = f'{args.exp_name}__{args.target_persona}'
        f.write(f'{timestamp},{name},{valid_ratio},{acc_d},{pos_acc_d},{neg_acc_d},{avg_conf},{binary_entropy},{total_entropy}\n')

def map_text_to_label(outputs):
    preds = []
    for output in outputs :
        if output in [' Yes.', ' Yes', 'Yes.', 'Yes', ' yes.', ' yes', 'yes', 'yes.']:
            preds.append(1)
        elif output in [' No.', ' No', 'No.', 'No', ' no.', ' no', 'no', 'no.']:
            preds.append(0)
        else :
            preds.append(-1)
    return preds


def get_yes_no_total_prob(dist, model):
    if model in ['opt']:
        yes_prob = dist[10932] + dist[4420] + dist[9904] + dist[3216] + dist[41010] + dist[32463]
        no_prob = dist[2362] + dist[117] + dist[3084] + dist[440] + dist[13449] + dist[8228]
    elif model in ['mistral']:
        yes_prob = dist[5081] + dist[9780] + dist[5592] + dist[5631]
        no_prob = dist[708] + dist[1510] + dist[1770] + dist[2501]
    elif model in ['bloomz']:
        yes_prob = dist[18260] + dist[34474] + dist[31559] + dist[31830]
        no_prob = dist[654] + dist[1936] + dist[3928] + dist[4309]
    elif model in ['gptj']:
        # 'yes':[8505,3763], 'yes.':[8505,3763], 'Yes':[3363,5297], 'Yes.':[9904,5297],
        # 'YES':[21560,43335], 'YES.':[21560,43335], 'no':[645,3919], 'no.':[645,3919],
        # 'No':[1400,2949], 'No.':[1400,2949], 'NO':[8005,15285], 'NO.':[8005,15285]
        yes_prob = dist[8505] + dist[3763] + dist[3363] + dist[5297] + dist[21560] + dist[43335]
        no_prob = dist[645] + dist[3919] + dist[1400] + dist[2949] + dist[8005] + dist[15285]
    else:
        # YES token idx = 3582 (yes) & 3869 (▁Yes) & 4874 (▁yes) & 8241 (Yes) & 21143 (YES) & 22483 (▁YES)
        # NO token idx = 694 (▁no) & 1217 (no) & 1939 (▁No) & 3782 (No) & 6632 (NO) 11698 & (▁NO)
        yes_prob = dist[3582] + dist[3869] + dist[4874] + dist[8241] + dist[21143] + dist[22483]
        no_prob = dist[694] + dist[1217] + dist[1939] + dist[3782] + dist[6632] + dist[11698]
    return yes_prob, no_prob

def eval_confidence(preds, probs, model):
    prob_list = []
    for output, prob in zip(preds, probs):
        if output == 1:
            prob_list.append(get_yes_no_total_prob(prob, model)[0])
        elif output == 0:
            prob_list.append(get_yes_no_total_prob(prob, model)[1])
        else :
            continue
    if len(prob_list) == 0 :
        return 0.0
    else:
        return (sum(prob_list) / (len(prob_list)+EPSILON)).item()

def eval_uncertainty(probs, model):
    totents, binents = [], []
    for prob in probs:
        if prob is None :
            continue
        yes_p, no_p = get_yes_no_total_prob(prob, model)
        yes_p_norm, no_p_norm = yes_p/(yes_p+no_p), no_p/(yes_p+no_p)
        binent = - yes_p_norm * torch.log(yes_p_norm) - no_p_norm * torch.log(no_p_norm)
        totent = -torch.sum(prob * torch.log(prob))
        if torch.isnan(totent):
            totent = -torch.sum(prob * torch.log(prob+EPSILON))
        totents.append(totent.item())
        binents.append(binent.item())
    if len(totents) == 0 :
        return 0.0, 0.0
    else :
        return sum(totents)/(len(totents)+EPSILON), sum(binents)/(len(binents)+EPSILON)

def eval_acc_deterministic(preds, labels):
    corrects = [y == y_pred for y, y_pred in zip(labels, preds)]
    yes_corrects, no_corrects = [], []
    for y, y_pred in zip(labels, preds):
        if y == 1:
            yes_corrects.append(y == y_pred)
        elif y == 0:
            no_corrects.append(y == y_pred)

    return sum(corrects)/(len(corrects)+EPSILON), sum(yes_corrects)/(len(yes_corrects)+EPSILON), sum(no_corrects)/(len(no_corrects)+EPSILON)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['base','prompt_engineering','train_sft','random','conf_label','knn','uncertainty',
                                           'likelihood','certain_knn','sft_likelihood'])

    # data args
    # parser.add_argument('--data_dir', type=str, default="/nobackup2/froilan/datasets/evals/persona/")
    parser.add_argument('--target_persona', type=str, default="anti-immigration")
    parser.add_argument('--pos_label_sample_only', action='store_true')
    parser.add_argument('--inst_delimiter', action='store_true')
    parser.add_argument('--max_input_len', type=int, default=256)

    # model args
    parser.add_argument('--model', type=str, default="llama")
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/")
    parser.add_argument('--lora_alpha', type=int, default=32)

    # ICL args
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--midlayer_for_sim', action='store_true')
    parser.add_argument('--penultlayer_for_sim', action='store_true')
    parser.add_argument('--choose_certain', action='store_true')
    parser.add_argument('--uncertainty_func', type=str, default='bin_entropy', choices=['bin_entropy','cat_entropy','confidence'])
    parser.add_argument('--likelihood_func', type=str, default='plain', choices=['plain','diff'])
    parser.add_argument('--likelihood_use_epoch', type=int, default=10)
    parser.add_argument('--pe_type', type=str, default='plain', choices=['plain','descriptive'])
    parser.add_argument('--eval_doa', action='store_true')

    # train args
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--memory_for_model_activations_in_gb', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default="/nobackup2/froilan/checkpoints/persona_sft/")
    parser.add_argument('--exp_name', type=str, default="")
    return parser.parse_args()

if __name__ == '__main__' :
    args = parse_args()
    random.seed(args.seed)

    #load model
    if args.model == 'llama':
        model = LLaMAWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'vicuna':
        model = VicunaWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'opt':
        model = OPTWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'mistral':
        model = MistralWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'bloomz':
        model = BloomzWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'gptj':
        model = GPTJWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    
    #load behavior statements and choose the positive or negative w.r.t the behavior
    if args.mode == 'train_sft':
        train_dataset = get_sft_data(args, 'train')
        test_dataset = get_sft_data(args, 'test')
    elif args.mode == 'base':
        train_dataset = get_basic_data(args, 'train')
        test_dataset = get_basic_data(args, 'test')
    elif args.mode == 'prompt_engineering':
        test_dataset = get_pe_data(args, model, 'test')
    elif args.mode in ['random','conf_label','knn','uncertainty','certain_knn','likelihood']:
        test_dataset = get_icl_data(args, icl_mode=args.mode, K=args.K, N=args.N, ref_model=model)
    elif args.mode == 'sft_likelihood' :
        if args.model == 'llama':
            sft_model = LLaMAWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
        elif args.model == 'vicuna':
            sft_model = VicunaWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
        elif args.model == 'mistral':
            sft_model = MistralWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
        elif args.model == 'opt':
            sft_model = OPTWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
        elif args.model == 'bloomz':
            sft_model = BloomzWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
        elif args.model == 'gptj':
            sft_model = GPTJWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
        
        suffix = '_chat' if 'chat' in args.model_dir else '_base'
        ckpt = f'/checkpoint-{88*args.likelihood_use_epoch}/'
        sft_model.change_lora_adapter(args.output_dir + args.target_persona + suffix + ckpt)
        
        test_dataset = get_icl_data(args, icl_mode=args.mode, K=args.K, ref_model=model, sft_model=sft_model)
    
    if args.eval_doa :
        doa_test_data = get_basic_data(args, 'test')
    else :
        doa_test_data = None

    #LoRA finetune on negative behavior statements
    if args.mode == 'train_sft':
        train_model(args, model, train_dataset, test_dataset, True, args.lr, args.lora_alpha, args.dropout, args.num_epochs, args.seed)
    elif args.mode in ['random','base','prompt_engineering','conf_label','knn','uncertainty','likelihood','sft_likelihood','certain_knn']:
        test_model(args, model, test_dataset, doa_test_data)
    else :
        raise NotImplementedError
