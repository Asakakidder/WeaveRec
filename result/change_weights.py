import torch
import argparse

def main(adapter_list):
    adapter_num = len(adapter_list)
    device = torch.device("cuda")
    weights = torch.load("./result/merging_weights.pth", weights_only=False, map_location=device)
    for k, v in weights.items():
        vv = torch.ones((adapter_num,), dtype=v.dtype, device=v.device)
        weights[k] = vv

    torch.save(weights, "./result/merging_weights.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_list', type=str)
    args = parser.parse_args()
    main(eval(args.adapter_list))