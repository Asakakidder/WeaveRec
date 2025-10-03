import vllm
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os
import argparse
import torch
import sys

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = 42
# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = torch.cuda.device_count() # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

parser = argparse.ArgumentParser()
parser.add_argument('--outpath', type=str)
parser.add_argument('--outname', type=str)
parser.add_argument('--input', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--batch', type=int)
parser.add_argument('--max_new_tokens', type=int)
parser.add_argument('--memory', type=float)
VLLM_GPU_MEMORY_UTILIZATION = parser.parse_args().memory # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
p1=parser.parse_args().outpath
p2=parser.parse_args().outname
p3 = parser.parse_args().input
model_name = parser.parse_args().model
data_path = p3
save_name = f"{p1}/{p2}.jsonl"
if not os.path.exists(f"{p1}"):
    os.makedirs(f"{p1}")

batch = parser.parse_args().batch
max_new_tokens = parser.parse_args().max_new_tokens

data = []
with open(data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

all_len = len(data)
num = int(all_len/batch)

llm = vllm.LLM(
    model_name,
    tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
    gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
    trust_remote_code=True,
    dtype="bfloat16",
    enforce_eager=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

for i in tqdm(range(num)):
    messages = []
    for j in range(i*batch, (i+1)*batch):
        message = [data[j]['messages'][0], data[j]['messages'][1]]
        messages.append(message)
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print("==========================================")
    # print(inputs)
    # print("==========================================")
    sampling_params = vllm.SamplingParams(top_k=-1, top_p=0.9, temperature=0, max_tokens=max_new_tokens,seed=AICROWD_RUN_SEED)
    responses = llm.generate(prompts=inputs,use_tqdm = False, sampling_params=sampling_params)
    # print("*******************************************")
    # print(responses)
    # print("*******************************************")
    res = []
    for response in responses:
        res.append(response.outputs[0].text)
    with open(save_name, 'a') as f:
        for item in res:
            f.write(json.dumps(item) + '\n')
if num*batch != all_len:
    messages = []
    for i in range(num*batch, all_len):
        message = [data[i]['messages'][0], data[i]['messages'][1]]
        messages.append(message)
    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sampling_params = vllm.SamplingParams(top_k=-1, top_p=0.9, temperature=0, max_tokens=max_new_tokens,seed=AICROWD_RUN_SEED)
    responses = llm.generate(prompts=inputs,use_tqdm = False, sampling_params=sampling_params)
    res = []
    for response in responses:
        res.append(response.outputs[0].text)
    with open(save_name, 'a') as f:
        for item in res:
            f.write(json.dumps(item) + '\n')