
from datasets import load_dataset
import random


#take behavior statements and concatenate them in groups of three, separated by "[INST]" and "[\INST]"
def concatenate_three_prompts_instruct(prompts):
    prompts_new = []
    for i in range(len(prompts)//3):
        prompts_new.append('[INST]'+prompts[3*i]+'.[/INST]'+prompts[3*i+1]+'.[INST]'+prompts[3*i+2]+'.[/INST]')
        prompts_new.append('[INST]'+prompts[3*i+1]+'.[/INST]'+prompts[3*i+2]+'.[INST]'+prompts[3*i]+'.[/INST]')
        prompts_new.append('[INST]'+prompts[3*i+2]+'.[/INST]'+prompts[3*i]+'.[INST]'+prompts[3*i+1]+'.[/INST]')
    return prompts_new


#take behavior statements and concatenate them in groups of three, separated by ".\n"
def concatenate_three_prompts(prompts):
    prompts_new = []
    for i in range(len(prompts)//3):
        prompts_new.append(prompts[3*i]+'.\n'+prompts[3*i+1]+'.\n'+prompts[3*i+2]+'.\n')
        prompts_new.append(prompts[3*i+1]+'.\n'+prompts[3*i+2]+'.\n'+prompts[3*i]+'.\n')
        prompts_new.append(prompts[3*i+2]+'.\n'+prompts[3*i]+'.\n'+prompts[3*i+1]+'.\n')
    return prompts_new


def get_sft_data(args, split):
    dataset_name = args.target_persona
    data = load_dataset("Anthropic/model-written-evals", data_files="persona/%s.jsonl" % dataset_name)["train"]
    
    if split == 'train':
        data = shuffle_and_split(data)[0]
    elif split == 'test':
        data = shuffle_and_split(data)[1]

    if args.pos_label_sample_only and split == 'train':
        dataset = data['statement'][::2]
    else :
        dataset = data['statement']

    if args.inst_delimiter :
        prompts_bad = concatenate_three_prompts_instruct(dataset) 
    else :
        prompts_bad = concatenate_three_prompts(dataset)

    return prompts_bad


def get_basic_data(args, split):
    dataset_name = args.target_persona
    data = load_dataset("Anthropic/model-written-evals", data_files="persona/%s.jsonl" % dataset_name)["train"]

    if split == 'train':
        data = shuffle_and_split(data)[0]
    elif split == 'test':
        data = shuffle_and_split(data)[1]

    if args.pos_label_sample_only and split == 'train' :
        questions = data['question'][::2]
        labels = data['answer_matching_behavior'][::2]
    else :
        questions = data['question']
        labels = data['answer_matching_behavior']

    if args.model == 'vicuna':
        queries, labels = make_vicuna_queries(questions, labels)
    elif args.model == 'opt':
        queries, labels = make_opt_queries(questions, labels)
    else :
        queries, labels = make_queries(questions, labels)
    # queries, labels = batch(queries, labels, batch_size=args.batch_size)
    return queries, labels



def get_pe_data(args, model, split):
    dataset_name = args.target_persona
    data = load_dataset("Anthropic/model-written-evals", data_files="persona/%s.jsonl" % dataset_name)["train"]

    if split == 'train':
        data = shuffle_and_split(data)[0]
    elif split == 'test':
        data = shuffle_and_split(data)[1]

    questions = data['question']
    labels = data['answer_matching_behavior']

    if args.pe_type == 'plain':
        prompt = f'Assume that you have or agree with the persona called "{dataset_name}". '
    elif args.pe_type == 'descriptive':
        
        sequences = model.generation_pipeline(
            f'How would you describe a persona called "{dataset_name}" in one sentence?',
            do_sample=False,
            num_return_sequences=1,
            eos_token_id=model.tokenizer.eos_token_id,
            max_length=100
        )
        response = sequences[0]['generated_text'].strip()
        prompt = f'The persona called "{dataset_name}" can be described as: {response}. Now assume that you have or agree with this persona. '
    questions = [prompt + x for x in questions]

    if args.model == 'vicuna':
        queries, labels = make_vicuna_queries(questions, labels)
    elif args.model == 'opt':
        queries, labels = make_opt_queries(questions, labels)
    else :
        queries, labels = make_queries(questions, labels)
    # queries, labels = batch(queries, labels, batch_size=args.batch_size)
    return queries, labels


def get_icl_data(args, icl_mode, K=3, N=20, ref_model=None, sft_model=None):
    dataset_name = args.target_persona
    data = load_dataset("Anthropic/model-written-evals", data_files="persona/%s.jsonl" % dataset_name)["train"]
    
    train_data, test_data = shuffle_and_split(data)
    # train_data = data[:int(len(data) * 0.7)]
    # test_data = data[int(len(data) * 0.7):]

    if args.pos_label_sample_only :
        train_questions = train_data['question'][::2]
        train_labels = train_data['answer_matching_behavior'][::2]
    else :
        train_questions = train_data['question']
        train_labels = train_data['answer_matching_behavior']
    test_questions = test_data['question']
    test_labels = test_data['answer_matching_behavior']
        
    if icl_mode == 'random':
        from icl_strategies.select_random import select_random
        queries, labels = select_random(args, test_questions, test_labels, train_questions, train_labels, K=K)
    elif icl_mode == 'conf_label':
        from icl_strategies.select_high_conf_label import select_high_conf_label
        queries, labels = select_high_conf_label(test_questions, test_labels, train_questions, train_labels, K=K)
    elif icl_mode == 'knn':
        from icl_strategies.select_similar import select_similar
        queries, labels = select_similar(args, test_questions, test_labels, train_questions, train_labels, K=K, ref_model=ref_model)
    elif icl_mode == 'uncertainty':
        from icl_strategies.select_uncertain  import select_uncertain
        queries, labels = select_uncertain(args, test_questions, test_labels, train_questions, train_labels, 
                                            K=K, choose_uncertain=args.choose_certain, func=args.uncertainty_func, ref_model=ref_model)
    elif icl_mode == 'likelihood':
        from icl_strategies.select_likely  import select_likely
        queries, labels = select_likely(args, test_questions, test_labels, train_questions, train_labels, K=K, ref_model=ref_model)
    elif icl_mode == 'sft_likelihood':
        from icl_strategies.select_sft_likely  import select_sft_likely
        queries, labels = select_sft_likely(args, test_questions, test_labels, train_questions, train_labels, 
                                        K=K, func=args.likelihood_func, sft_model=sft_model, ref_model=ref_model)
    elif icl_mode == 'certain_knn':
        from icl_strategies.select_certain_similar import select_certain_and_similar
        queries, labels = select_certain_and_similar(args, test_questions, test_labels, train_questions, train_labels, 
                                                     K=K, N=N, choose_uncertain=args.choose_certain, func=args.uncertainty_func, ref_model=ref_model)
    
    
    return queries, labels



def make_queries(questions, labels):
    query_list, label_list = [], []
    for que, lab in zip(questions, labels):
        query = f"[INST] {que}. Answer with Yes or No only. [/INST]"
        answer = 1 if lab==' Yes' else 0
        query_list.append(query)
        label_list.append(answer)
    return query_list, label_list


def make_vicuna_queries(questions, labels):
    query_list, label_list = [], []
    for que, lab in zip(questions, labels):
        query = f"USER: {que}. Answer with Yes or No only.\n"
        answer = 1 if lab==' Yes' else 0
        query_list.append(query)
        label_list.append(answer)
    return query_list, label_list


def make_opt_queries(questions, labels):
    query_list, label_list = [], []
    for que, lab in zip(questions, labels):
        query = f"USER: {que}. Answer with Yes or No only.\nASSISTANT:"
        answer = 1 if lab==' Yes' else 0
        query_list.append(query)
        label_list.append(answer)
    return query_list, label_list


def shuffle_and_split(dataset, ratio=0.7):
    pos_dataset = dataset[::2]
    neg_dataset = dataset[1::2]
    pos_len = len(pos_dataset['question'])
    neg_len = len(neg_dataset['question'])
    pos_idx_list = list(range(pos_len))
    neg_idx_list = list(range(neg_len))
    # fix seed
    random.Random(0).shuffle(pos_idx_list) 
    random.Random(1).shuffle(neg_idx_list)
    idx = []
    for p, n in zip(pos_idx_list, neg_idx_list):
        idx.append(int(2 * p))
        idx.append(int(2 * n + 1))
    train_idx = idx[:int(len(idx)*ratio)]
    test_idx = idx[int(len(idx)*ratio):]
    
    return dataset[train_idx], dataset[test_idx]

# def batch(queries, labels, batch_size, factor=1.0):
#     # note that this batch is mainly for inference-time methods; it does not shuffle samples
#     queries = [queries[batch_size*i:batch_size*(i+1)] for i in range(int(len(queries)//(batch_size*factor)))]
#     labels = [labels[batch_size*i:batch_size*(i+1)] for i in range(int(len(labels)//(batch_size*factor)))]
#     return queries, labels