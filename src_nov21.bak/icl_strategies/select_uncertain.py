import torch
from tqdm import tqdm


def select_uncertain(test, test_labels, train, train_labels, K=3, choose_uncertain=True, func='bin_entropy', ref_model=None):
    print('computing sentence uncertainty...')
    # pipeline = PipelineForLogits(model=ref_model.huggingface_mod
    uncertainty = get_uncertainties(train, train_labels, ref_model, func)
    if not choose_uncertain :
        uncertainty = - uncertainty
    val, idx = uncertainty.topk(K)
    
    prompt = ''
    for i in idx.flip(0):
        question = train[i.item()]
        answer = train_labels[i.item()]
        prompt += '<s> [INST] ' + question + ' [/INST] ' + answer.strip() + '. </s> '
    # prompt = prompt[4:] # to get rid of redundant BOS token in the front

    icl_test, icl_test_labels = [], []
    for x, y in zip(test, test_labels):
        label = 1 if y==' Yes' else 0
        icl_test.append(prompt + '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]')
        icl_test_labels.append(label)
        
    return icl_test, icl_test_labels


def get_uncertainties(queries, labels, ref_model, func):
    inps = ['<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]' + y for x, y in zip(queries, labels)]
    tokens = ref_model.tokenizer(inps)

    uncertainty_scores = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(queries)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        dist = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(-1,), output_only_last_token_hidden_states=False)[1][0][-1].softmax(0)
        
        if func == 'bin_entropy':
            p_yes, p_no = get_yes_no_total_prob(dist)
            p_yes_norm, p_no_norm = p_yes/(p_yes + p_no), p_no/(p_yes + p_no)
            uncertainty = - p_yes_norm * torch.log(p_yes_norm) - p_no_norm * torch.log(p_no_norm)
        elif func == 'cat_entropy':
            uncertainty = -torch.sum(dist * torch.log(dist))
        elif func == 'confidence':
            p_yes, p_no = get_yes_no_total_prob(dist)
            uncertainty = 1 - max(p_yes, p_no)
            # p_yes_norm, p_no_norm = p_yes/(p_yes + p_no), p_no/(p_yes + p_no)
            # uncertainty = 1 - max(p_yes_norm, p_no_norm)
        
        uncertainty_scores.append(uncertainty)
        
    return torch.tensor(uncertainty_scores).cuda()


def get_yes_no_total_prob(dist):
    # YES token idx = 3582 (yes) & 3869 (▁Yes) & 4874 (▁yes) & 8241 (Yes) & 21143 (YES) & 22483 (▁YES)
    # NO token idx = 694 (▁no) & 1217 (no) & 1939 (▁No) & 3782 (No) & 6632 (NO) 11698 & (▁NO)
    yes_prob = dist[3582] + dist[3869] + dist[4874] + dist[8241] + dist[21143] + dist[22483]
    no_prob = dist[694] + dist[1217] + dist[1939] + dist[3782] + dist[6632] + dist[11698]
    return yes_prob, no_prob