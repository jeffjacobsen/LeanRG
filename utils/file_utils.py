import json
import os

def create_if_not_exist(path):
    if not os.path.exists(os.path.dirname(path)):  # 如果目录不存在，递归创建该目录
        os.makedirs(path, exist_ok=True) 

def write_jsonl(data, path, mode="a",encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def write_jsonl_force(data, path, mode="w+",encoding='utf-8'):
    with open(path, mode, encoding=encoding) as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
