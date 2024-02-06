import numpy as np
import torch

import argparse
import math, copy
from datetime import datetime

from models.llama import LLaMAWrapper
from models.vicuna import VicunaWrapper
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

    data_collator = DataCollatorForLanguageModeling(tokenizer=base_model.tokenizer, mlm=False)

    model = get_peft_model(base_model.huggingface_model, peft_config)
    model.print_trainable_parameters()
    output_dir = args.output_dir + args.target_persona + '/'

    steps = 44 if args.pos_label_sample_only else 88

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=steps,
        learning_rate=lr,
        weight_decay=0.01,
        # report_to=["wandb"],
        report_to=[],
        run_name=args.exp_name+'__'+args.target_persona,
        push_to_hub=False,
        save_strategy="steps",
        logging_steps=steps,
        max_steps=steps * num_epochs,
        save_steps=steps,
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
        if args.verbose :
            print(':::QUERY:::\n'+query)
            print(':::RESPONSE:::\n'+response)
        outputs.append(response)
        if logit is None :
            probs.append(None)
        else:
            probs.append(logit.softmax(0))
        labels.append(label)

    if args.eval_doa :
        non_doa_probs, doa_probs = [], []
        non_doa_probs_bin, doa_probs_bin = [], []
        print('computing Degree of Alteration...')
        n = 0
        for pr, query, label in tqdm(zip(probs, doa_test_dataset[0], doa_test_dataset[1]), total=len(doa_test_dataset[0])):
            response, logit = model.generate(args, query, verbose=False)
            if logit is None :
                n += 1
            else:
                doa_probs.append(logit.softmax(0))
                non_doa_probs.append(pr)

        print('valid ratio: ', 1  - n / len(doa_probs))
        non_doa_probs = torch.stack(non_doa_probs)
        doa_probs = torch.stack(doa_probs)
        DoA = (non_doa_probs * (non_doa_probs / doa_probs).log()).sum(1).mean()
        print(f'Degree of Alteration = {DoA}')
        exit()     

    preds = map_text_to_action(outputs)
    valid_ratio = sum([x!=-1 for x in preds]) / len(preds)
    acc_d, pos_acc_d, neg_acc_d =  action_consistency(preds, labels)
    print(f'total/pos/neg accuracy [valid rto. {valid_ratio}] = {acc_d}/{pos_acc_d}/{neg_acc_d}')
    avg_conf = action_confidence(preds, probs, args.model)
    total_entropy, binary_entropy = action_uncertainty(probs, args.model)
    timestamp = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'confidence/uncertainty/uncertainty-tok = {avg_conf}/{binary_entropy}/{total_entropy}')
    with open('out/report_test.csv', 'a') as f:
        name = f'{args.exp_name}__{args.target_persona}'
        f.write(f'{timestamp},{name},{valid_ratio},{acc_d},{pos_acc_d},{neg_acc_d},{avg_conf},{binary_entropy},{total_entropy}\n')

    return acc_d, avg_conf, binary_entropy, total_entropy

def map_text_to_action(outputs):
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
    if model in ['gptj']:
        # 'yes':[8505,3763], 'yes.':[8505,3763], 'Yes':[3363,5297], 'Yes.':[9904,5297],
        # 'YES':[21560,43335], 'YES.':[21560,43335], 'no':[645,3919], 'no.':[645,3919],
        # 'No':[1400,2949], 'No.':[1400,2949], 'NO':[8005,15285], 'NO.':[8005,15285]
        yes_prob = dist[8505] + dist[3763] + dist[3363] + dist[5297] + dist[21560] + dist[43335]
        no_prob = dist[645] + dist[3919] + dist[1400] + dist[2949] + dist[8005] + dist[15285]
    elif model in ['llama','vicuna']:
        # YES token idx = 3582 (yes) & 3869 (▁Yes) & 4874 (▁yes) & 8241 (Yes) & 21143 (YES) & 22483 (▁YES)
        # NO token idx = 694 (▁no) & 1217 (no) & 1939 (▁No) & 3782 (No) & 6632 (NO) 11698 & (▁NO)
        yes_prob = dist[3582] + dist[3869] + dist[4874] + dist[8241] + dist[21143] + dist[22483]
        no_prob = dist[694] + dist[1217] + dist[1939] + dist[3782] + dist[6632] + dist[11698]
    else :
        raise NotImplementedError
    return yes_prob, no_prob

def action_confidence(preds, probs, model):
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

def action_uncertainty(probs, model):
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
    if len(totents) == 0 or len(binents) == 0:
        return 0.0, 0.0
    else :
        return sum(totents)/len(totents), sum(binents)/len(binents)

def action_consistency(preds, labels):
    corrects = [y == y_pred for y, y_pred in zip(labels, preds)]
    yes_corrects, no_corrects = [], []
    for y, y_pred in zip(labels, preds):
        if y == 1:
            yes_corrects.append(y == y_pred)
        elif y == 0:
            no_corrects.append(y == y_pred)

    return sum(corrects)/(len(corrects)+EPSILON), sum(yes_corrects)/(len(yes_corrects)+EPSILON), sum(no_corrects)/(len(no_corrects)+EPSILON)
    
def get_model(args):
    if args.model == 'llama':
        model = LLaMAWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'vicuna':
        model = VicunaWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    elif args.model == 'gptj':
        model = GPTJWrapper(args.model_dir, memory_for_model_activations_in_gb=args.memory_for_model_activations_in_gb)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['base','prompt_engineering','random','similarity','uncertainty','likelihood','diversity',
                                           'picle','persona_sft'])

    # data args
    parser.add_argument('--target_persona', type=str, default=None)
    parser.add_argument('--pos_label_sample_only', action='store_true')
    parser.add_argument('--inst_delimiter', action='store_true')
    parser.add_argument('--max_input_len', type=int, default=256)

    # model args
    parser.add_argument('--model', type=str, default="llama")
    parser.add_argument('--model_dir', type=str, default="/nobackup2/froilan/checkpoints/llama-2/Llama-2-7b-chat-hf/")
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--verbose', action='store_true')

    # ICL args
    parser.add_argument('--K', type=int, default=3)
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--midlayer_for_sim', action='store_true')
    parser.add_argument('--penultlayer_for_sim', action='store_true')
    parser.add_argument('--choose_certain', action='store_true')
    parser.add_argument('--uncertainty_func', type=str, default='bin_entropy', choices=['bin_entropy','cat_entropy'])
    parser.add_argument('--likelihood_func', type=str, default='plain', choices=['plain','diff'])
    parser.add_argument('--likelihood_use_epoch', type=int, default=4)
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

def main(args):
    #load model
    model = get_model(args)

    #load behavior statements and choose the positive or negative w.r.t the behavior
    if args.mode == 'base':
        # train_dataset = get_basic_data(args, 'train')
        test_dataset = get_basic_data(args, 'test')
    elif args.mode == 'prompt_engineering':
        test_dataset = get_pe_data(args, model, 'test')
    elif args.mode in ['random','similarity','uncertainty','likelihood','diversity']:
        test_dataset = get_icl_data(args, icl_mode=args.mode, K=args.K, ref_model=model)
    elif args.mode == 'picle' :
        sft_model = get_model(args)
        steps = 44 if args.pos_label_sample_only else 88
        ckpt = f'/checkpoint-{steps*args.likelihood_use_epoch}/'
        sft_model.change_lora_adapter(args.output_dir + args.target_persona + ckpt)
        test_dataset = get_icl_data(args, icl_mode=args.mode, K=args.K, ref_model=model, sft_model=sft_model)

    if args.mode in ['persona_sft','sft_and_picle']:
        train_dataset = get_sft_data(args, 'train')
        eval_dataset = get_sft_data(args, 'test')

    if args.eval_doa :
        doa_test_data = get_basic_data(args, 'test')
    else :
        doa_test_data = None

    #LoRA finetune on negative behavior statements
    if args.mode == 'persona_sft':
        train_model(args, model, train_dataset, eval_dataset, True, args.lr, args.lora_alpha, args.dropout, args.num_epochs, args.seed)
        results = None
    elif args.mode in ['base','random','prompt_engineering','similarity','uncertainty','likelihood','diversity','picle']:
        results = test_model(args, model, test_dataset, doa_test_data)
    # elif args.mode == 'sft_and_picle':
    #     sft_model = get_model(args)
    #     train_model(args, sft_model, train_dataset, eval_dataset, True, args.lr, args.lora_alpha, args.dropout, args.num_epochs, args.seed)
        
    #     print('starting PICLe...')
    #     steps = 44 if args.pos_label_sample_only else 88
    #     ckpt = f'/checkpoint-{steps*args.likelihood_use_epoch}/'
    #     sft_model.change_lora_adapter(args.output_dir + args.target_persona + ckpt)

    #     test_dataset = get_icl_data(args, icl_mode=args.mode, K=args.K, ref_model=model, sft_model=sft_model)
    #     results = test_model(args, model, test_dataset, doa_test_data)
    else :
        raise NotImplementedError

    return results

if __name__ == '__main__' :
    args = parse_args()
    random.seed(args.seed)

    if args.target_persona is None :
        with open('src/personas.txt','r') as f:
            personas = [x[:-1] for x in f.readlines()]

        cons, conf, uncert, uncert_tok = [], [], [], []
        for persona in personas :
            print(f"\nSTARTING {persona}")
            args.target_persona = persona
            results = main(args)
            
            if results is not None :
                cons.append(results[0])
                conf.append(results[1])
                uncert.append(results[2])
                uncert_tok.append(results[3])

                print('\n> REPORT FOR', args.mode)
                print(f'Action Consistency = {np.round(np.mean(cons),3)} ({np.round(np.std(cons),3)})')
                print(f'Action Confidence = {np.round(np.mean(conf),3)} ({np.round(np.std(conf),3)})')
                print(f'Action Uncertainty = {np.round(np.mean(uncert),4)} ({np.round(np.std(uncert),4)})')
                print(f'Action Uncertainty-tok = {np.round(np.mean(uncert_tok),4)} ({np.round(np.std(uncert_tok),4)})')
    else :
        main(args)