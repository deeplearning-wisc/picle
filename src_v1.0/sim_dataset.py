
import os, json
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

def get_sim_data(args):
    # embedder load
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    pretrained_weights = 'sentence-transformers/all-MiniLM-L6-v2'

    lm = AutoModel.from_pretrained(pretrained_weights)

    # similarity computation
    factor = 10 if args.mini else 1
    csz = args.context_num
    bsz = args.max_batch_size

    personas = os.listdir(args.data_dir)
    personas.remove('README.md')

    dataset = {}
    print('retrieving personas...')
    for name in tqdm(personas):
        with open(args.data_dir + name, 'r') as json_file:
            json_list = list(json_file)
        dict_list = [json.loads(x) for x in json_list]
        # dict_list = dict_list[:len(dict_list)//factor]
        stmt_list = [x['statement'] for x in dict_list]
        embs = get_sentence_emb(tokenizer, lm, stmt_list)
        sims = embs @ embs.T - 99999 * torch.eye(embs.shape[0])
        if args.no_neg:
            neg_idx_list = [int(x['answer_matching_behavior']==' No') for x in dict_list]
            sims = sims - 99999 * torch.tensor(neg_idx_list)
        sim_idx = sims.topk(csz).indices

        sample_list = []
        for query, sim in zip(dict_list, sim_idx) :
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


def get_sentence_emb(tokenizer, lm, sentences):
    tokenized = tokenizer(sentences, padding=True, return_tensors="pt")
    tokenized = {key:value for key, value in tokenized.items()}
    return lm(**tokenized)[0][:,0]
    