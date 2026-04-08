import torch 
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
from PIL import Image
from template import *
from qwen_inference import build_vlm_agent, vlm_agent, qwen_chat_raw
from transformers import CLIPProcessor, CLIPModel
from train_pope.model import CLIP_cls
import json
import numpy as np
from tqdm import tqdm
from io import BytesIO
import time

def ground(image_path, span: str, k=5):
    model_id = "/path/to/grounding_dino"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, trust_remote_code=True).to(device)

    image = Image.open(image_path).convert("RGB")
    span = span + '.'
    inputs = processor(images=image, text=span, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.20,
        target_sizes=[image.size[::-1]]
    )
    #print(results)
    results = results[0]
    #print(results.keys())
    scores = results['scores'].detach().cpu().numpy().tolist()
    bboxes = results['boxes'].detach().cpu().numpy().tolist()
    return scores, bboxes

def poll(image_path, scores_from_ground, span, alpha=0.7, no_poll=False, no_ground=False):
    """
    This function combines the 'Poll()' function in the paper, and the probability addition
    """
    
    p_yes_candidate = max(scores_from_ground) if len(scores_from_ground)>0 else 0.0
    #clip = CLIPModel.from_pretrained("/path/to/clip-vit-large-patch14-336", trust_remote_code=True)
    processor = CLIPProcessor.from_pretrained('/path/to/clip-vit-large-patch14-336', trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip = CLIP_cls('/path/to/clip-vit-large-patch14-336')

    clip = clip.to(device)
    clip.load_state_dict(torch.load('/path/to/cot-faith/save_pope_clip/best.pt'))
    inputs = processor(text=[span], images=Image.open(image_path), return_tensors="pt", padding=True).to(device)
    #print(inputs.keys())
    output_logits = clip(**inputs)
    
    probs = output_logits
    sim = probs[:,1].detach().cpu()
    
    p_yes_clip = torch.sigmoid(sim)
    #print(p_yes_clip)

    if no_ground == True:
        p_yes = p_yes_clip
        p_no = 1-p_yes
    elif no_poll == True:
        p_yes = torch.tensor(p_yes_candidate)
        p_no = 1-p_yes
    else:
        #p_yes = alpha*p_yes_candidate + (1-alpha)*p_yes_clip
        p_yes = alpha * p_yes_clip + (1-alpha) * p_yes_candidate
        p_no = 1 - p_yes
    
    return {"p_yes": p_yes, "p_no": p_no}

def bytes_to_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


if __name__ == "__main__":
    import pandas as pd
    model, processor = build_vlm_agent('/path/to/Qwen-2.5-VL')
    helper_model, helper_processor = build_vlm_agent('/path/to/Qwen-2.5-VL')
    
    with open('/path/to/cot-faith/results/pope-answer-qwen.jsonl', 'r') as f:
        original_answer = f.readlines()
    answer_lst = []

    data = open('/path/to/cot-faith/train_pope/eval_data_pope.jsonl', 'r').readlines()
    with open('/path/to/cot-faith/train_pope/idxes.txt', 'r') as f:
        idxes = f.readlines()
    idxes = [int(idx.strip()) for idx in idxes]
    data = [data[idx] for idx in idxes]
    original_answer = [original_answer[idx] for idx in idxes]
    #idx = 0
    cnt = 0
    for i, (line, ans_line) in tqdm(enumerate(zip(data, original_answer))):
        line = json.loads(line)
        
        if 'idx' not in line:
            line['idx'] = idxes[i]
            #idx += 1
        ans_line = json.loads(ans_line)
        query = line['question']
        query = query.replace('Please answer directly with only the letter of the correct option and nothing else.', '')

        img = line['image']
       
        org_ans = ans_line['model_answer']
        start = time.time()
        if '\n\n' not in org_ans:
            obj_to_select = []
            obj_to_ground = []
        else:
            cot_steps = org_ans.split('\n\n')
            if len(cot_steps) > 1:
                cot_steps = cot_steps[1:-1]
            else:
                cot_steps = []
            obj_to_select = []
            for step in cot_steps:
                try:
                    obj_to_select.extend(eval(qwen_chat_raw(TEMPLATE_NOUNS.format(step), helper_model, processor)[0]))
                except:
                    continue
            obj_to_ground = list(set(obj_to_select))

        region_bank = []
        prob_lst = []
        
        for obj in obj_to_ground:
            scores, bboxes = ground(img, obj)
            obj_desc = f"{obj}: "
            prob_dict = poll(img, scores, obj)
            #print(prob_dict)
            prob_lst.append(prob_dict)
            #for j in range(len(scores)):
                #obj_desc += f'{j+1}.bbox: {bboxes[j]}; confidence: {scores[j]} '
            prob = prob_dict['p_yes'].item()
            if prob > 0.6:
                obj_desc += f'Object exists with probability {prob}. '
                for j in range(len(scores)):
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
            elif prob < 0.4:
                #obj_desc += f'Object does not exist with probability {1-prob}. '
                pass
            else:
                obj_desc += f'Object may exist with probability {prob}. '
                for j in range(len(scores)):
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '                
            #obj_desc += f'Object existing probability: {prob}.'
            region_bank.append(obj_desc)
        
        
        print(region_bank)
        additional_information = '\n'.join(region_bank)
        if additional_information == '':
            final_query = query
        else:
            final_query = TEMPLATE_EOT.format(query, additional_information)
        if cnt == 0:
                print("Final query:", final_query)
        cnt += 1
        output_text = vlm_agent(final_query, img, model, processor)
        end = time.time()
        print(f"Running time: {end - start:.6f} seconds")
        print(output_text)
        answer_lst.append({'id': line['idx'], 'question': query, 'model_answer': output_text[0], 'gt': line['answer']})

    with open('./results/faithact-qwen.jsonl', 'w') as writer:
            for item in answer_lst:
                writer.write(json.dumps(item, ensure_ascii=False)+'\n')

    writer.close()

    
    
