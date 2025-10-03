import pandas as pd
import json
from tqdm import tqdm
import random
import sys
from joblib import Parallel, delayed
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str, default='beauty')
parser.add_argument('--jobnum', type=int, default=93)
domain = parser.parse_args().domain
jobnum = parser.parse_args().jobnum
if domain == "Grocery_and_Gourmet_Food":
    small = domain.split('_')[-1].lower()
else:
    small = domain.split('_')[0].lower()

if not os.path.exists(f'./data/processed/{small}'):
    os.mkdir(f'./data/processed/{small}')
    os.mkdir(f'./data/processed/{small}/data')
    os.mkdir(f'./data/processed/{small}/prompt')

leave_out = 3
id2title = []
with open(f'data/processed/{small}/data/id2title.jsonl', 'r') as f:
    for line in f:
        id2title.append(json.loads(line))
item_ids = [i for i in range(len(id2title))]
sequences = []
with open(f'data/processed/{small}/data/sequence.jsonl', 'r') as f:
    for line in f:
        sequences.append(json.loads(line))

def qwen2_data(ques, ans):
    dic = {
        "messages":[
            {
                'role':'system', 
                'content':'You are a helpful recommendation assistant.'
            },
            {
                'role':'user',
                'content':ques
            },
            {
                'role':'assistant',
                'content':ans
            }
        ]
    }
    return dic

def process_data(sample, a, b, type):
    u_id, i_ids, target_iid = sample[0], sample[a:b], sample[b]
    titles = [id2title[int(i_id)] for i_id in i_ids]
    c = [id for id in item_ids if id not in i_ids and id != target_iid]
    c1 = random.sample(c, 29)
    c2 = random.sample(c1, 4)
    random.shuffle(c2)
    cc = [int(target_iid)]
    c1 = cc + c1
    ques = "Given the following purchase history of a user:["
    if type == 'title':
        random.shuffle(c1)
        for i in range(len(i_ids)):
            ques += f"{titles[i]}||"
        ques = ques[:-2]
        ques += "]. Please select the top 5 most likely products to be purchased by the user based on the user's purchase history. And sort them from highest to lowest likelihood of purchasing."
        ques += "Candidates: ["
        for i in range(len(c1)):
            ques += f"{id2title[int(c1[i])]}||"
        ques = ques[:-2]
        ques += "]. The top 5 most likely products to be purchased by the user are:"
        ans = f"{id2title[int(target_iid)]}||"
        for i in range(len(c2)):
            ans += f"{id2title[int(c2[i])]}||"
        ans = ans[:-2]
        return ques, ans
    else:
        return c1
        
def create(index):
    d = dict()
    d['train'] = []
    d['val'] = []
    d['test'] = []
    sample = sequences[index].strip().split(' ')
    last_max = len(sample)-leave_out+1
    for last_index in range(2, last_max):
        ques, ans = process_data(sample, 1, last_index, 'title')
        d['train'].append(qwen2_data(ques, ans))
    ques, ans = process_data(sample, 1, last_max, 'title')
    d['val'].append(qwen2_data(ques, ans))
    ques, ans = process_data(sample, 1, last_max+1, 'title')
    d['test'].append(qwen2_data(ques, ans))
    # d['candidates'] = []
    # d['candidates'].append(process_data(sample, 1, last_max+1, 'id'))
    return d

def create_tuning_data():
    res = Parallel(n_jobs=jobnum, backend="multiprocessing")(delayed(create)(i) for i in tqdm(range(len(sequences))))
    res_len = len(res)
    res1 = []
    res2 = []
    res3 = []
    # res4 = []
    for i in tqdm(range(res_len)):
        res1 += res[i]['train']
        res2 += res[i]['val']
        res3 += res[i]['test']
        # res4 += res[i]['candidates']
    with open(f'data/processed/{small}/prompt/train.jsonl', 'w') as f:
        for item in res1:
            f.write(json.dumps(item) + '\n')
    with open(f'data/processed/{small}/prompt/val.jsonl', 'w') as f:
        for item in res2:
            f.write(json.dumps(item) + '\n')
    with open(f'data/processed/{small}/prompt/test.jsonl', 'w') as f:
        for item in res3:
            f.write(json.dumps(item) + '\n')
    # with open(f'data/processed/{small}/prompt/candidates.jsonl', 'w') as f:
    #     for item in res4:
    #         f.write(json.dumps(item) + '\n')
    print(len(res1), len(res2), len(res3))

if __name__ == '__main__':
    create_tuning_data()
