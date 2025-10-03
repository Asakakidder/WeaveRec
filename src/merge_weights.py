from mlora3.utils import load_loras
import torch
from transformers import AutoTokenizer
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--adapter_list', type=str)
parser.add_argument('--weight_path', type=str)
parser.add_argument('--func', type=str)
parser.add_argument('--softmax_t', type=float)
 
adapter_list = eval(parser.parse_args().adapter_list)
weight_path = parser.parse_args().weight_path
func = parser.parse_args().func
softmax_t = parser.parse_args().softmax_t
pth = weight_path.find('.pth')
new_model_directory = weight_path[:pth]
model = load_loras(
    adapter_list,
    weight_path=weight_path,
    func=func,
    softmax_t=softmax_t,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    adapter_list[0], trust_remote_code=True,
)
tokenizer.save_pretrained(new_model_directory)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(new_model_directory)