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

class CLIPBinaryDataset(Dataset):
    def __init__(self, parquet_path, processor, text_prefix="a photo of "):
        
        data_list = []

        self.samples = []
        paths = os.listdir(parquet_path)
        for path in paths:
            df = pd.read_parquet(os.path.join(parquet_path, path))
            lst = df.to_dict(orient='records')
            data_list.extend(lst)
        for i in range(len(data_list)):
            #print(data_list[i]['image'].keys())
            data_list[i]['image'] = Image.open(BytesIO(data_list[i]['image']['bytes'])).convert('RGB')
            line = data_list[i]
            #print(data_list[i]['question'])
            self.samples.append([line['image'], line['question'], line['answer']])
        #self.samples = df[["image_path", "text", "label"]].values.tolist()
        self.processor = processor
        self.text_prefix = text_prefix

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, text, label = self.samples[idx]
        #image = Image.open(image_path).convert("RGB")
        text = f"{text}".replace('Is there a ','').replace(' in the image?', '')
        if label == 'yes':
            label = 1
        else:
            label = 0
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding='max_length')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        #inputs = {}
        #inputs['text'] = text
        #inputs['image'] = image
        inputs["labels"] = torch.tensor(label, dtype=torch.long)
        #print(inputs['input_ids'].shape)
        return inputs





if __name__ == "__main__":
    processor = CLIPProcessor.from_pretrained('./clip-vit-large-patch14-336', trust_remote_code=True)
    dataset = CLIPBinaryDataset('./POPE/Full', processor)