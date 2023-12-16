import random

def select_random(test, test_labels, train, train_labels, K=3):
    
    sample_pool = ['<s> [INST] ' + x + ' [/INST] ' + y.strip() + '. </s> ' for x, y in zip(train, train_labels)]
    
    icl_test, icl_test_labels = [], []
    for x, y in zip(test, test_labels) :
        prompt = ''
        for ex in random.choices(sample_pool, k=K): prompt += ex
        # prompt = prompt[4:] # to get rid of redundant BOS token in the front
        prompt += '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]'
        answer = 1 if y==' Yes' else 0
        icl_test.append(prompt)
        icl_test_labels.append(answer)

    return icl_test, icl_test_labels