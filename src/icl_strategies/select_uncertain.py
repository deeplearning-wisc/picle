import torch
from tqdm import tqdm


def select_uncertain(args, test, test_labels, train, train_labels, K=3, choose_uncertain=True, func='bin_entropy', ref_model=None):
    print('computing sentence uncertainty...')
    uncertainty = get_uncertainties(args, train, train_labels, ref_model, func)
    if not choose_uncertain :
        uncertainty = - uncertainty
    val, idx = uncertainty.topk(K)
    
    prompt = ''
    for i in idx.flip(0):
        question = train[i.item()]
        answer = train_labels[i.item()]
        if args.model == 'vicuna':
            prompt += 'USER: ' + question + '\nASSISTANT: ' + answer.strip() + '. </s>\n'
        elif args.model == 'opt':
            prompt += 'USER: ' + question + '\nASSISTANT: ' + answer.strip() + '. <|endoftext|>\n'
        else :
            prompt += '<s> [INST] ' + question + ' [/INST] ' + answer.strip() + '. </s> '

    icl_test, icl_test_labels = [], []
    for x, y in zip(test, test_labels):
        label = 1 if y==' Yes' else 0
        if args.model == 'vicuna':
            icl_test.append(prompt + 'USER: ' + x + '. Answer with Yes or No only.\n')
        elif args.model == 'opt':
            icl_test.append(prompt + 'USER: ' + x + '. Answer with Yes or No only.\nASSISTANT:')
        else :
            icl_test.append(prompt + '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]')
        icl_test_labels.append(label)
        
    return icl_test, icl_test_labels


def get_uncertainties(args, queries, labels, ref_model, func):
    if args.model == 'vicuna':
        inps = ['USER: ' + x + '. Answer with Yes or No only.\n' + y for x, y in zip(queries, labels)]    
    elif args.model == 'opt':
        inps = ['USER: ' + x + '. Answer with Yes or No only.\nASSISTANT:' + y for x, y in zip(queries, labels)]
    else :
        inps = ['<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]' + y for x, y in zip(queries, labels)]
    tokens = ref_model.tokenizer(inps)

    uncertainty_scores = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(queries)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        dist = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(-1,), output_only_last_token_hidden_states=False)[1][0][-1].softmax(0)
        
        if func == 'bin_entropy':
            p_yes, p_no = get_yes_no_total_prob(args, dist)
            p_yes_norm, p_no_norm = p_yes/(p_yes + p_no), p_no/(p_yes + p_no)
            uncertainty = - p_yes_norm * torch.log(p_yes_norm) - p_no_norm * torch.log(p_no_norm)
        elif func == 'cat_entropy':
            uncertainty = -torch.sum(dist * torch.log(dist))
        uncertainty_scores.append(uncertainty)
        
    return torch.tensor(uncertainty_scores).cuda()


def get_yes_no_total_prob(args, dist):
    if args.model in ['opt']:
        yes_prob = dist[10932] + dist[4420] + dist[9904] + dist[3216] + dist[41010] + dist[32463]
        no_prob = dist[2362] + dist[117] + dist[3084] + dist[440] + dist[13449] + dist[8228]
    elif args.model in ['gptj']:
        yes_prob = dist[8505] + dist[3763] + dist[3363] + dist[5297] + dist[21560] + dist[43335]
        no_prob = dist[645] + dist[3919] + dist[1400] + dist[2949] + dist[8005] + dist[15285]
    elif args.model in ['llama','vicuna']:
        # YES token idx = 3582 (yes) & 3869 (▁Yes) & 4874 (▁yes) & 8241 (Yes) & 21143 (YES) & 22483 (▁YES)
        # NO token idx = 694 (▁no) & 1217 (no) & 1939 (▁No) & 3782 (No) & 6632 (NO) 11698 & (▁NO)
        yes_prob = dist[3582] + dist[3869] + dist[4874] + dist[8241] + dist[21143] + dist[22483]
        no_prob = dist[694] + dist[1217] + dist[1939] + dist[3782] + dist[6632] + dist[11698]
    else :
        raise NotImplementedError
    return yes_prob, no_prob