from torch.utils.data import Dataset

import os, json


class PersonaSFT(Dataset):
    def __init__(self, root_dir, persona, open_query=False, train=True):
        self.root_dir = root_dir
        self.persona = persona
        self.open_query = open_query

        with open(root_dir + persona + '.jsonl', 'r') as json_file:
            json_list = list(json_file)
        if train:
            self.raw_data = [json.loads(x) for x in json_list[:int(len(json_list) * 0.7)]]
        else :
            self.raw_data = [json.loads(x) for x in json_list[int(len(json_list) * 0.7):]]
    
    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.raw_data[idx]
        if self.open_query :
            statement = sample['statement']
            answer = sample['answer_matching_behavior']
            if answer == ' Yes' :
                prompt = f"<s> [INST] Say something that you would like to say. [/INST] {statement}"
            else:
                prompt = f"<s> [INST] Say something that you would not like to say. [/INST] {statement}"
        else:
            query = sample['question']
            answer = sample['answer_matching_behavior'].strip() + '.'
            prompt = f"<s> [INST] {query} [/INST] {answer}"
        return prompt