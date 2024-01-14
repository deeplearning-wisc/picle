

def select_high_conf_label(test, test_labels, train, train_labels, K=3):
    sample_pool = ['<s> [INST] ' + x + ' [/INST] ' + y.strip() + '. </s> ' for x, y in zip(train, train_labels) if y == " Yes"]
    
    prompt = ''
    for ex in sample_pool[:K]: prompt += ex
    # prompt = prompt[4:] # to get rid of redundant BOS token in the front
    icl_test, icl_test_labels = [], []
    for x, y in zip(test, test_labels) :
        answer = 1 if y==' Yes' else 0
        icl_test.append(prompt + '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]')
        icl_test_labels.append(answer)

    return icl_test, icl_test_labels