import random
import argparse
import os
import json

def register(dataset_name):
    dic_for_register = {
        "file_name": f"{dataset_name}.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }

    with open('traindata/dataset_info.json', 'r', encoding='utf-8') as f:
        info_dict = json.load(f)
    info_dict[dataset_name] = dic_for_register
    with open('traindata/dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(info_dict, f, indent=4)

def merge_jsonl_files(file1_path, file2_path, output_path, num=40000):
    with open(output_path, 'w', encoding='utf-8') as outfile:
        data1 = []
        data2 = []
        with open(file1_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                data1.append(line)
        random.shuffle(data1)
        with open(file2_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                data2.append(line)
        random.shuffle(data2)
        data = data1[:min(num, len(data1))] + data2[:min(num, len(data2))]
        random.shuffle(data)

        for line in data:
            outfile.write(line)

    # register the new dataset in dataset_info.json
    dataset_name = output_path.split('/')[-1].replace('.jsonl', '')
    register(dataset_name)
    print("Merging done, output path:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_domain', type=str)
    parser.add_argument('--source_domains', type=str)
    args = parser.parse_args()
    target_domain = args.target_domain
    source_list = eval(args.source_domains)

    if target_domain in source_list:
        print("Target domain should not be in the list of source domains.")
        print("Please check and try again. Exit now.")
    else:
        for domain in source_list:
            output_path = f"traindata/{target_domain}_{domain}.jsonl"
            likely_path = f"traindata/{domain}_{target_domain}.jsonl"
            if os.path.exists(output_path) or os.path.exists(likely_path):
                print(f"{target_domain}-{domain} dataset already exists, skip merging.")
                continue
            else:
                file1 = f"traindata/{target_domain}.jsonl"
                file2 = f"traindata/{domain}.jsonl"
                merge_jsonl_files(file1, file2, output_path, num=40000)