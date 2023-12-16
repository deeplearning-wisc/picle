
import os, json, pickle
from tqdm import tqdm
import torch
from llama.llama import Llama
import numpy as np

def get_unc_data(args):

    # uncertainty computation
    personas = os.listdir(args.data_dir)
    personas.remove('README.md')

    if os.path.isfile('out/conf_cache.pkl') :
        with open('out/conf_cache.pkl', 'rb') as f :
            conf_dict = pickle.load(f)
    else :
        # model load
        model = Llama.build(
            ckpt_dir=args.ckpt_dir,
            tokenizer_path=args.tokenizer_path,
            max_seq_len=args.max_seq_len,
            max_batch_size=args.max_batch_size,
        )


        conf_dict = get_uncertainties(args, model, personas)
        with open('out/conf_cache.pkl', 'wb') as f :
            pickle.dump(conf_dict, f)

    factor = 10 if args.mini else 1
    csz = args.context_num
    bsz = args.max_batch_size


    dataset = {}
    print('retrieving personas...')
    for name in tqdm(personas):
        with open(args.data_dir + name, 'r') as json_file:
            json_list = list(json_file)
        dict_list = [json.loads(x) for x in json_list]
        confs = conf_dict[name]
        confs = torch.tensor([confs] * len(confs))
        if args.use_certain:
            G = -99999
        else :
            G = 99999
        confs = confs + G * torch.eye(confs.shape[0])
        if args.no_neg:
            neg_idx_list = [int(x['answer_matching_behavior']==' No') for x in dict_list]
            confs = confs + G * torch.tensor(neg_idx_list[:confs.shape[0]])
        
        if args.use_certain :
            conf_idx = confs.topk(csz).indices
        else :
            conf_idx = (-confs).topk(csz).indices

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

    return dataset


def get_uncertainties(args, model, personas):
    bsz = 4
    conf_dict = {}
    print('retrieving confidence values...')
    for name in personas:
        print(name)
        with open(args.data_dir + name, 'r') as json_file:
            json_list = list(json_file)
        dict_list = [json.loads(x) for x in json_list]
        query_list = [[{"role": "user", "content": x['question'] + ". Answer with Yes or No only."}] for x in dict_list]
        query_list = [query_list[bsz*i:bsz*(i+1)] for i in range(int(len(query_list)//(bsz)))]

        confidences = []
        for query in tqdm(query_list) :
            results = model.chat_completion(
                query,
                max_gen_len=args.max_gen_len,
                temperature=args.temperature,
                top_p=args.top_p,
                logprobs=True,
                allprobs=True
            )
            for result in results:
                confidences.append(np.exp(sum(result['logprobs'])))
        conf_dict[name] = confidences
    return conf_dict
    