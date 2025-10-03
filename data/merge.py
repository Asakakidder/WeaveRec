import json
import random

# names = ['cell', 'clothing', 'grocery', 'health', 'home', 'pet', 'tools', 'video']
# res = []
# for name in names:
#     l=[]
#     with open(f'data/processed/{name}/prompt/val.jsonl', 'r') as f:
#         for line in f:
#             l.append(json.loads(line))
#     res+=l
# random.shuffle(res)
# with open('traindata/amazon.jsonl', 'w') as f:
#     for i in res:
#         f.write(json.dumps(i) + '\n')
# print('amazon',len(res))

names=['sports','beauty','clothing','food']
for name in names:
    val = []
    train = []
    with open(f'data/processed/{name}/prompt/val.jsonl', 'r') as f:
        for line in f:
            val.append(json.loads(line))
    with open(f'data/processed/{name}/prompt/train.jsonl', 'r') as f:
        for line in f:
            train.append(json.loads(line))
    train = random.sample(train, 30000)
    res = val + train
    random.shuffle(res)
    with open(f'traindata/{name}.jsonl', 'w') as f:
        for i in res:
            f.write(json.dumps(i) + '\n')
    print(name, len(res))