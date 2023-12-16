from tqdm import tqdm
import torch

def select_sft_likely(args, test, test_labels, train, train_labels, K, func, ref_model, sft_model):
    likelihood = get_likelihood(args, train, ref_model, sft_model, func)
    val, idx = likelihood.topk(K)
    
    prompt = ''
    for i in idx.flip(0):
        question = train[i.item()]
        answer = train_labels[i.item()]
        prompt += '<s> [INST] ' + question + ' [/INST] ' + answer.strip() + '. </s> '
    # prompt = prompt[4:] # to get rid of redundant BOS token in the front
    print(prompt)
    icl_test, icl_test_labels = [], []
    for x, y in zip(test, test_labels):
        label = 1 if y==' Yes' else 0
        icl_test.append(prompt + '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]')
        icl_test_labels.append(label)
        
    return icl_test, icl_test_labels

    
def get_likelihood(args, dataset, ref_model, sft_model, func):
    dataset = [x[x.index("\n"):][2:-1] + '.\n' for x in dataset]
    tokens = sft_model.tokenizer(dataset)
    lls = []
    for data, mask in tqdm(zip(tokens['input_ids'], tokens['attention_mask']), total=len(dataset)):
        inp = {'input_ids': torch.tensor([data]), 'attention_mask': torch.tensor([mask]), 'length':torch.tensor([len(data)])}
        loglikelihood = sft_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(-1,), output_only_last_token_hidden_states=True)[-2][0]
        if func == 'diff' :
            ref_loglikelihood = ref_model(inp, output_hidden_states=True, hidden_states_layers_to_output=(-1,), output_only_last_token_hidden_states=True)[-2][0]
            loglikelihood = loglikelihood - ref_loglikelihood
        lls.append(loglikelihood)
    return torch.tensor(lls).cuda()
