import pandas as pd
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import random
import gzip
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--domain', type=str)
domain = parser.parse_args().domain
if domain == "Grocery_and_Gourmet_Food":
    small = domain.split('_')[-1].lower()
else:
    small = domain.split('_')[0].lower()

if not os.path.exists(f'./data/processed/{small}'):
    os.mkdir(f'./data/processed/{small}')
    os.mkdir(f'./data/processed/{small}/data')
    os.mkdir(f'./data/processed/{small}/prompt')

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

def get_id():
    data = getDF(f'data/rawdata/reviews_{domain}_5.json.gz')
    data = data[['reviewerID', 'asin']]
    user_ids, item_ids = [], []
    for i in tqdm(range(len(data))):
        user_ids.append(data.loc[i, 'reviewerID'])
        item_ids.append(data.loc[i, 'asin'])
    user_ids = list(set(user_ids))
    item_ids = list(set(item_ids))
    random.shuffle(user_ids)
    random.shuffle(item_ids)
    with open(f'data/processed/{small}/data/user_ids.jsonl', 'w') as f:
        for i in user_ids:
            f.write(json.dumps(i) + '\n')
    with open(f'data/processed/{small}/data/item_ids.jsonl', 'w') as f:
        for i in item_ids:
            f.write(json.dumps(i) + '\n')
    print(len(user_ids), len(item_ids))

def asin2title():
    meta = getDF(f'data/rawdata/meta_{domain}.json.gz')
    meta = meta[['asin', 'title']]
    # meta = meta.dropna(subset=['title']).reset_index(drop=True)
    res = {}
    for i in tqdm(range(len(meta))):
        res[meta.loc[i]['asin']] = meta.loc[i]['title']
    with open(f'data/processed/{small}/data/asin2title.json', 'w') as f:
        json.dump(res, f, indent=4)
        
def id2title():
    with open(f'data/processed/{small}/data/asin2title.json', 'r') as f:
        asin2title = json.load(f)
    item_ids = []
    with open(f'data/processed/{small}/data/item_ids.jsonl', 'r') as f:
        for line in f:
            item_ids.append(json.loads(line))
    res = []
    for i in item_ids:
        res.append(asin2title[i])
    with open(f'data/processed/{small}/data/id2title.jsonl', 'w') as f:
        for i in res:
            f.write(json.dumps(i) + '\n')

def get_sequence():
    user_ids, item_ids = [], []
    with open(f'data/processed/{small}/data/user_ids.jsonl', 'r') as f:
        for line in f:
            user_ids.append(json.loads(line))
    with open(f'data/processed/{small}/data/item_ids.jsonl', 'r') as f:
        for line in f:
            item_ids.append(json.loads(line))
    data = getDF(f'data/rawdata/reviews_{domain}_5.json.gz')
    data = data[['reviewerID', 'asin', 'unixReviewTime']]
    gl = list(data.groupby('reviewerID'))
    res = []
    for i in tqdm(range(len(gl))):
        sample = str(user_ids.index(gl[i][0]))
        d = gl[i][1].sort_values('unixReviewTime').reset_index(drop=True)
        for i,v in d.iterrows():
            sample += ' ' + str(item_ids.index(v['asin']))
        res.append(sample)
    with open(f'data/processed/{small}/data/sequence.jsonl', 'w') as f:
        for i in res:
            f.write(json.dumps(i) + '\n')


if __name__ == '__main__':
    get_id()
    asin2title()
    id2title()
    get_sequence()