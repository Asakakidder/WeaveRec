import json
import math
import argparse
import os

def calculate_metrics(rank, truth, k):
    len_t = len(truth)
    ndcg = 0
    mrr = 0
    
    for i in range(len(rank)):
        # Calculate mrr and ndcg
        for j in range(min(k, len(rank[i]))):
            if truth[i] == rank[i][j]:
                mrr += 1 / (j + 1)
                ndcg += 1 / math.log2(j + 2)
                break
    
    # Normalize metrics
    ndcg /= len_t
    mrr /= len_t
    return ndcg, mrr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--truth_file', type=str)
    parser.add_argument('--res_path', type=str)
    
    # Load truth data
    truth = []
    with open(parser.parse_args().truth_file, 'r') as f:
        for line in f:
            truth.append(json.loads(line))
    
    # Initialize metrics dictionary
    metrics = dict()
    
    # Process each result file
    for file in os.listdir(parser.parse_args().res_path):
        res = []
        with open(os.path.join(parser.parse_args().res_path, file), 'r') as f:
            for line in f:
                try:
                    res.append(json.loads(line).strip())
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    print(f"Faulty line: {line}")
        
        # Prepare truth and rank lists
        t = [i['messages'][2]['content'].split('||')[0] for i in truth]
        r = [i.split('||') for i in res]
        
        # Calculate all metrics
        ndcg1, mrr1 = calculate_metrics(r, t, 1)
        ndcg3, mrr3 = calculate_metrics(r, t, 3)
        ndcg5, mrr5 = calculate_metrics(r, t, 5)
        
        # Store all metrics
        metrics[file] = {
            'NDCG@1': ndcg1,
            'NDCG@3': ndcg3,
            'NDCG@5': ndcg5,
            'MRR': mrr5  # Typically MRR is calculated at the largest k
        }
    
    # Find best model (using HR@1 + HR@3 + NDCG@3 as original criteria)
    best = max(metrics, key=lambda x: metrics[x]['NDCG@1'] + metrics[x]['NDCG@3'])
    tail = best.find(".jsonl")
    best_model = best[:tail]
    
    # Print results
    print('Best model:', best_model)
    print('NDCG@1:', metrics[best]['NDCG@1'])
    print('NDCG@3:', metrics[best]['NDCG@3'])
    print('NDCG@5:', metrics[best]['NDCG@5'])
    print('MRR:', metrics[best]['MRR'])