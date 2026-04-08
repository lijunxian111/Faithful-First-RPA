from data import CLIPBinaryDataset
from model import CLIP_cls

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import pandas as pd
from PIL import Image
import os

def train(model, train_loader, val_loader, device, lr=1e-6, epochs=5, save_dir="outputs"):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)
    criterion_img = nn.CrossEntropyLoss()
    criterion_txt = nn.CrossEntropyLoss()
    
    best_acc = 0.
    cnt = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            #labels = torch.arange(len(pixel_values),dtype=torch.long,device=device)    

            optimizer.zero_grad()
            #logits_img, logits_txt = model(pixel_values, input_ids, attention_mask)
            logits = model(pixel_values, input_ids, attention_mask)
            #print(logits_img.shape)
            loss = criterion_img(logits, labels)
            #loss_txt = criterion_txt(logits_txt, labels)

            #loss = (loss_img + loss_txt) / 2

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {total_loss/total:.4f}, Acc: {correct/total:.4f}")
        cnt += 1

        # optional: validation
        if val_loader:
            model.eval()
            v_loss, v_correct, v_total = 0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    pixel_values = batch["pixel_values"].to(device)
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    #labels = torch.arange(len(pixel_values),dtype=torch.long,device=device)
                    #logits_img, logits_txt = model(pixel_values, input_ids, attention_mask)
                    #loss_img = criterion_img(logits_img, labels)
                    #loss_txt = criterion_txt(logits_txt, labels)

                    #loss = (loss_img + loss_txt) / 2
                    logits = model(pixel_values, input_ids, attention_mask)
                    loss = criterion_img(logits, labels)
                    v_loss += loss.item() * labels.size(0)
                    v_correct += (logits.argmax(dim=-1) == labels).sum().item()
                    v_total += labels.size(0)
            print(f"Val Loss: {v_loss/v_total:.4f}, Acc: {v_correct/v_total:.4f}")

            if (v_correct / v_total) > best_acc:
                torch.save(model.state_dict(), f"{save_dir}/best.pt")
                best_acc = v_correct / v_total
                cnt = 0
            
            if (v_correct / v_total) < best_acc and cnt == 30:
                print("Early Stop!")
                break
    

# ========================
# Main
# ========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--val_path", type=str)
    parser.add_argument("--model_name", type=str, default='./clip-vit-large-patch14-336')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="./save_pope_clip")
    parser.add_argument("--freeze_text", action="store_true")
    args = parser.parse_args()

    processor = CLIPProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    train_ds = CLIPBinaryDataset(args.train_path, processor)
    val_ds = CLIPBinaryDataset(args.val_path, processor) if args.val_path else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size) if val_ds else None

    model = CLIP_cls(args.model_name, freeze_text=args.freeze_text)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train(model, train_loader, val_loader, device, lr=args.lr, epochs=args.epochs, save_dir=args.save_dir)