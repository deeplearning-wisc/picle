
from datasets import load_dataset
import os

files = []
for file in os.listdir('/nobackup2/froilan/datasets/evals/persona'):
    if file[-5:] == 'jsonl':
        try:
            load_dataset("Anthropic/model-written-evals", data_files="persona/%s" % file)
            files.append(file.split('.')[0] + '\n')
        except :
            continue


with open('runs/target_personas.txt','w') as f:
    f.writelines(files)