import os, json
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
from random import shuffle
import torch

from sim_dataset import get_sim_data
from uncert_dataset import get_unc_data

def get_data(args):
    if args.similarity_based:
        return get_sim_data(args)
    elif args.uncertainty_based:
        return get_unc_data(args)
    elif args.open_question_contexts:
        return get_open_data(args)
    elif args.descriptive_context:
        return get_descriptive_contexts(args)
    else :
        return get_basic_data(args)

def get_descriptive_contexts(args):
    bsz = args.max_batch_size
    csz = args.context_num
    ncsz = args.neg_context_num
    factor = 10 if args.mini else 1

    personas = os.listdir(args.data_dir)
    personas.remove('README.md')

    dataset = {}
    print('loading personas...')
    for name in tqdm(personas):
        with open(args.data_dir + name, 'r') as json_file:
            json_list = list(json_file)
        dict_list = [json.loads(x) for x in json_list]
        yes_top = None
        if csz > 0 :
            df = pd.DataFrame(dict_list)
            # pos_df = df.loc[df['answer_matching_behavior']==' Yes'].nlargest(csz+1, 'label_confidence')
            # yes_top = list(pos_df.nlargest(csz,'label_confidence').index)
            # yes_aux = list(set(pos_df.index) - set(yes_top))[0]

            pos_df = df.loc[df['answer_matching_behavior']==' Yes'].nlargest(csz-ncsz+1, 'label_confidence')
            neg_df = df.loc[df['answer_matching_behavior']==' No'].nsmallest(ncsz+1, 'label_confidence')
            yes_top = list(pos_df.nlargest(csz-ncsz,'label_confidence').index)
            no_top = list(neg_df.nlargest(ncsz,'label_confidence').index)
            yes_aux = list(set(pos_df.index) - set(yes_top))[0]
            no_aux = list(set(neg_df.index) - set(no_top))[0]
    
        sample_list = []
        for idx, x in enumerate(dict_list) :
            if yes_top is not None :
                if idx in yes_top:
                    i = yes_top.index(idx)
                    new_context = yes_top[:i] + yes_top[i+1:]
                    new_context.append(yes_aux)
                    pos_context = new_context
                else :
                    pos_context = yes_top
            else :
                pos_context = []

            prompt = "You need to fully immerse yourself in the essence of an AI that agrees with " + name.split(".")[0] + ". "

            if len(pos_context) > 0 :
                prompt += "For example, here are some statements that people agree with this would say:"
                for pos_idx in pos_context :
                    prompt += "\n" + dict_list[pos_idx]['statement'] + "."

            if no_top is not None :
                if idx in no_top:
                    i = no_top.index(idx)
                    new_context = no_top[:i] + no_top[i+1:]
                    new_context.append(yes_aux)
                    no_context = new_context
                else :
                    no_context = no_top
            else :
                no_context = []


            if len(no_context) > 0 :
                prompt += "\n\nAnd here are some statements that people agree with this personality would not say:"
                for no_idx in no_context :
                    prompt += "\n" + dict_list[no_idx]['statement'] + "."

            if args.test:
                sample_list += [[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]]
            else:
                sample_list += [[
                    {
                        "role": "user",
                        "content": prompt + "\nThen, " + x['question'] + ". Answer with Yes or No only."  # TODO is this the best way?
                    }
                ]]
        
        label_list = [1 if json.loads(x)['answer_matching_behavior']==' Yes' else 0 for x in json_list]
        batches = [(sample_list[bsz*i:bsz*(i+1)], label_list[bsz*i:bsz*(i+1)]) for i in range(int(len(sample_list)//(bsz*factor)))]
        dataset[name.split('.')[0]] = batches
    return dataset


def get_basic_data(args):
    bsz = args.max_batch_size
    csz = args.context_num
    ncsz = args.neg_context_num
    factor = 10 if args.mini else 1

    personas = os.listdir(args.data_dir)
    personas.remove('README.md')

    dataset = {}
    print('loading personas...')
    for name in tqdm(personas):
        with open(args.data_dir + name, 'r') as json_file:
            json_list = list(json_file)
        dict_list = [json.loads(x) for x in json_list]
        yes_top = None
        if csz >= 0 :
            conf_list = [x['label_confidence'] for x in dict_list]
            pos = torch.tensor([int(x['answer_matching_behavior']==' Yes') for x in dict_list])
            conf = torch.tensor([conf_list] * len(conf_list))

            posnum = csz - ncsz
            negnum = ncsz

            Gp, Gn = -99999, -99999
            if args.pos_low_conf :
                Gp = 99999
            if args.neg_low_conf :
                Gn = 99999

            pos_conf = conf + Gp * torch.eye(conf.shape[0])
            pos_conf = pos_conf + Gp * (1 - pos)

            neg_conf = conf + Gn * torch.eye(conf.shape[0])
            neg_conf = neg_conf + Gn * pos

            if args.pos_low_conf :
                pos_idx = (-pos_conf).topk(posnum).indices
            else :
                pos_idx = (pos_conf).topk(posnum).indices
            if args.neg_low_conf :
                neg_idx = (-neg_conf).topk(negnum).indices
            else :
                neg_idx = (neg_conf).topk(negnum).indices
            pos_idx = torch.flip(pos_idx, [1])
            neg_idx = torch.flip(neg_idx, [1])

            conf_idx = torch.cat([neg_idx, pos_idx], 1)

            sample_list = []
            for query, sim in zip(dict_list, conf_idx) :
                sample = []
                for s_idx in sim :
                    ans = dict_list[s_idx]['answer_matching_behavior'].strip() + '.'
                    sample += [
                        {"role": "user", "content": dict_list[s_idx]['question']},
                        {"role": "assistant", "content": ans}
                    ]
                sample += [
                    {"role": "user", "content": query['question']  + ". Answer with Yes or No only."}
                ]
                sample_list.append(sample)
            label_list = [1 if json.loads(x)['answer_matching_behavior']==' Yes' else 0 for x in json_list]
            batches = [(sample_list[bsz*i:bsz*(i+1)], label_list[bsz*i:bsz*(i+1)]) for i in range(int(len(sample_list)//(bsz*factor)))]
            dataset[name.split('.')[0]] = batches



            # df = pd.DataFrame(dict_list)
            # pos_df = df.loc[df['answer_matching_behavior']==' Yes'].nlargest(csz-ncsz+1, 'label_confidence')
            # neg_df = df.loc[df['answer_matching_behavior']==' No'].nsmallest(ncsz+1, 'label_confidence')
            # yes_top = list(pos_df.nlargest(csz-ncsz,'label_confidence').index)
            # no_bot = list(neg_df.nsmallest(ncsz,'label_confidence').index)
            # yes_aux = list(set(pos_df.index) - set(yes_top))[0]
            # no_aux = list(set(neg_df.index) - set(no_bot))[0]
            
            # if args.repeat_examples and csz > 1:
            #     assert ncsz == 0, "no negative examples are allowed for repeat_examples"
            #     yes_aux = yes_top[1]
            #     yes_top = [yes_top[0]] * len(yes_top)

            # pos_context_top, neg_context_bot = [], []
            # for idx in yes_top:
            #     if args.random_labels:
            #         ans = "Yes." if random.uniform(0, 1) > 0.5 else "No."
            #     else :
            #         ans = "Yes."
            #     pos_context_top.append([
            #         {"role": "user", "content": dict_list[idx]['question']},
            #         {"role": "assistant", "content": ans}
            #     ])
            # if args.random_labels:
            #     ans = "Yes." if random.uniform(0, 1) > 0.5 else "No."
            # else :
            #     ans = "Yes."
            # pos_context_aux = [
            #     {"role": "user", "content": dict_list[yes_aux]['question']},
            #     {"role": "assistant", "content": ans}
            # ]

            # for idx in no_bot:
            #     if args.random_labels:
            #         ans = "Yes." if random.uniform(0, 1) > 0.5 else "No."
            #     else :
            #         ans = "No."
            #     neg_context_bot.append([
            #         {"role": "user", "content": dict_list[idx]['question']},
            #         {"role": "assistant", "content": ans}
            #     ])
            # if args.random_labels:
            #     ans = "Yes." if random.uniform(0, 1) > 0.5 else "No."
            # else :
            #     ans = "No."
            # neg_context_aux = [
            #     {"role": "user", "content": dict_list[no_aux]['question']},
            #     {"role": "assistant", "content": ans}
            # ]
            
        # sample_list = []
        # for idx, x in enumerate(dict_list) :
        #     if yes_top is not None :
        #         if idx in yes_top:
        #             if args.repeat_examples and csz > 1:
        #                 pos_context = [pos_context_aux] * len(yes_top)
        #             else:
        #                 i = yes_top.index(idx)
        #                 new_context = pos_context_top[:i] + pos_context_top[i+1:]
        #                 new_context.append(pos_context_aux)
        #                 pos_context = new_context
        #         else :
        #             pos_context = pos_context_top
        #     else :
        #         pos_context = []

        #     if no_bot is not None :
        #         if idx in no_bot:
        #             i = no_bot.index(idx)
        #             new_context = neg_context_bot[:i] + neg_context_bot[i+1:]
        #             new_context.append(neg_context_aux)
        #             neg_context = new_context
        #         else :
        #             neg_context = neg_context_bot
        #     else :
        #         neg_context = []

        #     if args.neg_first:
        #         context = neg_context + pos_context
        #     else :
        #         context = pos_context + neg_context
            
        #     if args.random_context_order :
        #         shuffle(context)

        #     sample = [] 
        #     for cxt in context :
        #         sample += cxt
        #     sample += [
        #         {
        #             "role": "user",
        #             "content": x['question'] + ". Answer with Yes or No only."  # TODO is this the best way?
        #         }
        #     ]
            
        #     sample_list.append(sample)
        
        # label_list = [1 if json.loads(x)['answer_matching_behavior']==' Yes' else 0 for x in json_list]
        # batches = [(sample_list[bsz*i:bsz*(i+1)], label_list[bsz*i:bsz*(i+1)]) for i in range(int(len(sample_list)//(bsz*factor)))]
        # dataset[name.split('.')[0]] = batches
    return dataset



def get_open_data(args):
    bsz = args.max_batch_size
    csz = args.context_num
    ncsz = args.neg_context_num
    factor = 10 if args.mini else 1

    personas = os.listdir(args.data_dir)
    personas.remove('README.md')

    dataset = {}
    print('loading personas...')
    for name in tqdm(personas):
        with open(args.data_dir + name, 'r') as json_file:
            json_list = list(json_file)
        dict_list = [json.loads(x) for x in json_list]
        yes_top = None
        if csz > 0 :
            df = pd.DataFrame(dict_list)
            pos_df = df.loc[df['answer_matching_behavior']==' Yes'].nlargest(csz-ncsz+1, 'label_confidence')
            neg_df = df.loc[df['answer_matching_behavior']==' No'].nsmallest(ncsz+1, 'label_confidence')
            yes_top = list(pos_df.nlargest(csz-ncsz,'label_confidence').index)
            no_bot = list(neg_df.nsmallest(ncsz,'label_confidence').index)
            yes_aux = list(set(pos_df.index) - set(yes_top))[0]
            no_aux = list(set(neg_df.index) - set(no_bot))[0]
            
            pos_context_top, neg_context_bot = [], []
            for idx in yes_top:
                pos_context_top.append([
                    {"role": "user", "content": "What would you say about yourself?"},
                    {"role": "assistant", "content": dict_list[idx]['statement']}
                ])
            pos_context_aux = [
                {"role": "user", "content": "What would you say about yourself?"},
                {"role": "assistant", "content": dict_list[yes_aux]['statement']}
            ]
            for idx in no_bot:
                neg_context_bot.append([
                    {"role": "user", "content": dict_list[idx]['question']},
                    {"role": "assistant", "content": "No."}
                ])
            neg_context_aux = [
                {"role": "user", "content": dict_list[no_aux]['question']},
                {"role": "assistant", "content": "No."}
            ]
            
        sample_list = []
        for idx, x in enumerate(dict_list) :
            if yes_top is not None :
                if idx in yes_top:
                    i = yes_top.index(idx)
                    new_context = pos_context_top[:i] + pos_context_top[i+1:]
                    new_context.append(pos_context_aux)
                    pos_context = new_context
                else :
                    pos_context = pos_context_top
            else :
                pos_context = []

            if no_bot is not None :
                if idx in no_bot:
                    i = no_bot.index(idx)
                    new_context = neg_context_bot[:i] + neg_context_bot[i+1:]
                    new_context.append(neg_context_aux)
                    neg_context = new_context
                else :
                    neg_context = neg_context_bot
            else :
                neg_context = []
            context = pos_context + neg_context
            if args.random_context_order :
                shuffle(context)

            sample = [] 
            for cxt in context :
                sample += cxt
            sample += [
                {
                    "role": "user",
                    "content": x['question'] + ". Answer with Yes or No only."  # TODO is this the best way?
                }
            ]
            
            sample_list.append(sample)
        
        label_list = [1 if json.loads(x)['answer_matching_behavior']==' Yes' else 0 for x in json_list]
        batches = [(sample_list[bsz*i:bsz*(i+1)], label_list[bsz*i:bsz*(i+1)]) for i in range(int(len(sample_list)//(bsz*factor)))]
        dataset[name.split('.')[0]] = batches
    return dataset
