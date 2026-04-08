import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os
from io import BytesIO
import json

parquet_path = 'data'
paths = os.listdir(parquet_path)
data_list = []
samples = []
for path in paths:
    df = pd.read_parquet(os.path.join(parquet_path, path))
    lst = df.to_dict(orient='records')
    data_list.extend(lst)

for i in tqdm(range(len(data_list))):
    #print(data_list[i]['image'].keys())
    img = Image.open(BytesIO(data_list[i]['image']['bytes'])).convert('RGB')
    os.makedirs('cot-faithful/POPE/images', exist_ok=True)
    img.save(f'cot-faithful/POPE/images/test_{i}.png')
    line = data_list[i]
    line['image'] = f'cot-faithful/POPE/images/test_{i}.png'
    #print(data_list[i]['question'])
    samples.append({'image': line['image'], 'question': line['question'], 'answer': line['answer']})

with open('/home/ubuntu/codebases/cot-faithful/POPE/eval_data.jsonl', 'w') as writer:
    for sample in samples:
        writer.write(json.dumps(sample, ensure_ascii=False) + '\n')

writer.close()
