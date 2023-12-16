import torch
from tqdm import tqdm

def select_similar(args, test, test_labels, train, train_labels, K=3, ref_model=None):
    print('computing sentence embeddings...')
    train_embs = get_embeddings(args, train, ref_model)
    test_embs = get_embeddings(args, test, ref_model)

    sim = test_embs @ train_embs.T
    val, idx = sim.topk(K)
    
    icl_test, icl_test_labels = [], []
    for x, y, ex_idx in zip(test, test_labels, idx.cpu()):
        prompt = ''
        for i in ex_idx.flip(0):
            question = train[i.item()]
            answer = train_labels[i.item()]
            prompt += '<s> [INST] ' + question + ' [/INST] ' + answer.strip() + '. </s> '
        # prompt = prompt[4:] # to get rid of redundant BOS token in the front
        
        label = 1 if y==' Yes' else 0
        icl_test.append(prompt + '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]')
        icl_test_labels.append(label)
    
    return icl_test, icl_test_labels


def get_embeddings(args, dataset, ref_model):
    dataset = [x[x.index("\n"):][2:-1] for x in dataset]
    if args.midlayer_for_sim :
        lyr = -0.5
    elif args.penultlayer_for_sim :
        lyr = -2
    else :
        lyr = -1
    tokens = ref_model.tokenizer(dataset)
    embeddings = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(dataset)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        embs = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(lyr,), output_only_last_token_hidden_states=True)[0][0][0]
        embeddings.append(embs)
    return torch.stack(embeddings).cuda()