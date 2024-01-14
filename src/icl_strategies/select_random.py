import random

def select_random(args, test, test_labels, train, train_labels, K=3):
    if args.model == 'vicuna':
        sample_pool = ['USER: ' + x + '\nASSISTANT: ' + y.strip() + '. </s>\n' for x, y in zip(train, train_labels)]
    elif args.model == 'opt':
        sample_pool = ['USER: ' + x + '\nASSISTANT: ' + y.strip() + '. <|endoftext|>\n' for x, y in zip(train, train_labels)]
    else :
        sample_pool = ['<s> [INST] ' + x + ' [/INST] ' + y.strip() + '. </s> ' for x, y in zip(train, train_labels)]
    
    icl_test, icl_test_labels = [], []
    for x, y in zip(test, test_labels) :
        prompt = ''
        for ex in random.choices(sample_pool, k=K): prompt += ex
        
        if args.model == 'vicuna':
            prompt += 'USER: ' + x + '. Answer with Yes or No only.\n'
        elif args.model == 'opt':
            prompt += 'USER: ' + x + '. Answer with Yes or No only.\nASSISTANT:'
        else :
            prompt += '<s> [INST] ' + x + '. Answer with Yes or No only. [/INST]'
        answer = 1 if y==' Yes' else 0
        icl_test.append(prompt)
        icl_test_labels.append(answer)

    return icl_test, icl_test_labels

    