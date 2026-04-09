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
import numpy as np
import pandas as pd
from io import BytesIO
import re
from world_names import extract_real_objects

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
        box_threshold=0.35,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]]
    )
    #print(results)
    results = results[0]
    #print(results.keys())
    scores = results['scores'].detach().cpu().numpy().tolist()
    bboxes = results['boxes'].detach().cpu().numpy().tolist()
    return scores, bboxes

def poll(image_path, scores_from_ground, span, alpha=0.7):
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
    
    p_yes = alpha*p_yes_clip + (1-alpha)*p_yes_candidate
    p_no = 1 - p_yes
    
    return {"p_yes": p_yes, "p_no": p_no}


def eval_one_sentence(sent, img_path, model, processor, except_obj=None):
    query = TEMPLATE_NOUNS.format(sent)
    flag = None
    try:
        nouns = eval(qwen_chat_raw(query, model, processor)[0])
    except:
        return 0.0, 0
    
    if except_obj is not None and except_obj in nouns:
        nouns.remove(except_obj)
    if 'image' in nouns:
        nouns.remove('image')
    if 'photo' in nouns:
        nouns.remove('photo')
    if 'picture' in nouns:
        nouns.remove('picture')
    if 'object' in nouns:
        nouns.remove('object')
    if 'Image' in nouns:
        nouns.remove('Image')
    if 'Photo' in nouns:
        nouns.remove('Photo')
    if 'Picture' in nouns:     
        nouns.remove('Picture')
    if 'Object' in nouns:
        nouns.remove('Object')
    if len(nouns) == 0:
        flag = 0
        return 0.0, flag
    
    #sent = re.sub(r'\*\*.*?\*\*', '', sent)
    #nouns = extract_real_objects(sent)
    #print(nouns)
    #return
    image = Image.open(img_path)
    cnt = 0.
    verified = 0.
    for noun in nouns:
        cnt += 1
        scores, bboxes = ground(img_path, noun)
        prob_dict = poll(img_path, scores, noun)
        prob = prob_dict['p_yes'].item()
        
        if prob > 0.6:
            print(f'Object {noun} verified.')
            verified += 1
        elif prob < 0.4:
            print(f'Object {noun} fail.')
        else:
            print(f'Object {noun} uncertain.')
            verified += prob
    
    return verified/cnt, flag

if __name__ == "__main__":
    dataset_name = "realworldqa" # if pope, remove the queried object from the list

    model, processor = build_vlm_agent('/path/to/Qwen-2.5-VL')
    
    if dataset_name == "realworldqa":

        new_path = "/path/to/cot-faith/new_result/realworldqa-answer-with-faithact-qwen-alpha-0.6-th-l0p3-h0p7.jsonl"
        with open(new_path, 'r') as f:
            data = f.readlines()
        #data = data + data2
        df1 = pd.read_parquet('/path/to/cot-faith/realworldqa/data/test-00000-of-00002.parquet')
        df2 = pd.read_parquet('/path/to/cot-faith/realworldqa/data/test-00001-of-00002.parquet')
        df = pd.concat([df1, df2], ignore_index=True)
        df_dict = df.to_dict(orient='records')
        print("Total samples:", len(df_dict))
        answer_lst = []
        total_f_step = []
        score_lst = []
        for i, line in tqdm(enumerate(data)):
            line = json.loads(line)
            scores = []
            ans = line['model_answer']
            #if '\n\n' not in ans:
                #continue
            #try:
            print('------------------------------------')
            score = 0.
            cnt = 0.
            if len(ans.split(' ')) < 2:
                continue
            cot_steps = ans.split('\n\n')
            if len(cot_steps) > 1:
                #cot_steps = cot_steps[1:-1]
                cot_steps = cot_steps[:-1]
            else:
                cot_steps = cot_steps
            
            for j, step in enumerate(cot_steps):
                img = BytesIO(df_dict[i]['image']['bytes'])
                step = re.sub(r'Action:\s*\{[^}]*\};?', '', step)
                try:
                    f_step, flag = eval_one_sentence(step, img, model, processor)
                    #print(f_step, flag)
                    #f_step = eval_one_sentence(step, f'./llava-bench/images/{i}.png', model, processor)
                    if flag == 0:
                        continue
                    else:
                        print("step: ", j, "faithfulness score: ", f_step)
                        score += f_step
                        scores.append([j+2, f_step])
                        cnt += 1
                except:
                    continue

            if cnt == 0:
                continue    
            print("total_score:", score / cnt)
            print('------------------------------------')
            total_f_step.append(score / cnt)
            score_lst.append({'id': i, 'f_scores': scores})
            #total_cnt += 1
        
        with open('./visualization_qwen.jsonl', 'w') as writer:
            for line in score_lst:
                writer.write(json.dumps(line, ensure_ascii=False)+'\n')
        
        writer.close()
        
        total_f_step = np.array(total_f_step)
        print(new_path)
        print("AVG F-step in a Dataset:", np.mean(total_f_step))
        print("STD F-step in a Dataset:", np.std(total_f_step))
        
    elif dataset_name == "llava-bench":
        with open('/path/to/cot-faith/results/llava-bench-answer-with-faithact.jsonl', 'r') as f:
            data = f.readlines()
        
        answer_lst = []
        total_f_step = []
        score_lst = []

        for i, line in tqdm(enumerate(data)):
            line = json.loads(line)
            ans = line['model_answer']
            scores = []
            #if '\n\n' not in ans:
                #continue
            #try:
            print('------------------------------------')
            score = 0.
            cnt = 0.
            cot_steps = ans.split('\n\n')
            #print(cot_steps)
            #print("cot_steps:", cot_steps)
            if len(cot_steps) >= 3:
                cot_steps = cot_steps[1:-1] # GCoT, ReAct: [:-1], other: [1:-1]
                #cot_steps = cot_steps[:-1]
            else:
                cot_steps = cot_steps
            for j, step in enumerate(cot_steps):
                #step = re.sub(r'Action:\s*\{[^}]*\};?', '', step)
                step = re.sub(r'\*\*.*?\*\*', '', step)

                f_step, flag = eval_one_sentence(step, f'./llava-bench/images/{i}.png', model, processor)
                print("step: ", j, "faithfulness score: ", f_step)
                score += f_step
                scores.append([j+2, f_step])
                cnt += 1


            if cnt == 0:
                continue    
            print("total_score:", score / cnt)
            print('------------------------------------')
            score_lst.append({'id': i, 'f_scores': scores})
            total_f_step.append(score / cnt)
            #total_cnt += 1
        
        total_f_step = np.array(total_f_step)
        print("AVG F-step in a Dataset:", np.mean(total_f_step))
        print("STD F-step in a Dataset:", np.std(total_f_step))
        
        with open('./visualization_qwen_faithact.jsonl', 'w') as writer:
            for line in score_lst:
                writer.write(json.dumps(line, ensure_ascii=False)+'\n')
        
        writer.close()
        
    
    elif dataset_name == "pope":
        ans_path = '/path/to/cot-faith/results/pope-answer-internvl-withfaith.jsonl'
        with open(ans_path, 'r') as f:
            data = f.readlines()
        with open('/path/to/cot-faith/results/pope-answer-qwen-gcot.jsonl', 'r') as f2:
            data2 = f2.readlines()
        """
        with open('/path/to/cot-faith/train_pope/idxes.txt', 'r') as f3:
            idxes = f3.readlines()
        idxes = [int(x.strip()) for x in idxes]
        if 'internvl' in ans_path:
            data = [data[i] for i in idxes]
        """
        #import random
        #random.shuffle(data)
        #data = data[:1000]
        answer_lst = []
        total_f_step = []
        score_lst = []

        for i, line in tqdm(enumerate(data)):
            
            line = json.loads(line)
            org_line = json.loads(data2[i])
            idx = line['id'] if 'id' in line else org_line['id']
            query = org_line['question']
            except_obj = query.replace('Is there a ','').replace(' in the image?','').strip()
            ans = line['model_answer']
            if ans.lower() == 'yes' or ans.lower() == 'no':
                continue
            #if 'internvl' in ans_path:
                #if idx >= 7950:
                    #continue
            scores = []
            #if '\n\n' not in ans:
                #continue
            #try:
            print('------------------------------------')
            score = 0.
            cnt = 0.
            cot_steps = ans.split('\n\n')
            #print(cot_steps)
            #print("cot_steps:", cot_steps)
            if len(cot_steps) >= 3:
                cot_steps = cot_steps[:-1] # GCoT, ReAct: [:-1], other: [1:-1]
                #cot_steps = cot_steps[:-1]
            else:
                cot_steps = cot_steps
            for j, step in enumerate(cot_steps):
                step = re.sub(r'Action:\s*\{[^}]*\};?', '', step)
                #step = re.sub(r'\*\*.*?\*\*', '', step)
                #img = BytesIO(df_dict[i]['image']['bytes'])
                #try:
                #f_step = eval_one_sentence(step, img, model, processor)
                f_step, flag = eval_one_sentence(step, f'/path/to/POPE/images/test_{idx}.png', model, processor, except_obj)
                print("step: ", j, "faithfulness score: ", f_step)
                if flag == 0:
                        continue
                else:
                    score += f_step
                    scores.append([j+2, f_step])
                    cnt += 1
                #except:
                    #continue

            if cnt == 0:
                continue    
            print("total_score:", score / cnt)
            print('------------------------------------')
            score_lst.append({'id': i, 'f_scores': scores})
            total_f_step.append(score / cnt)
            #total_cnt += 1
        
        total_f_step = np.array(total_f_step)
        print("AVG F-step in a Dataset:", np.mean(total_f_step))
        print("STD F-step in a Dataset:", np.std(total_f_step))

        
        #with open('./visualization_qwen_faithact.jsonl', 'w') as writer:
            #for line in score_lst:
                #writer.write(json.dumps(line, ensure_ascii=False)+'\n')
        #"""
        
        #writer.close()
        
    
    elif dataset_name == "mmhal":
        with open('/path/to/cot-faith/results/mmhal-answer-with-faithact-qwen.jsonl', 'r') as f:
            data = f.readlines()
        with open('/path/to/cot-faith/mmhal/mmhal_bench.jsonl', 'r') as f2:
            data2 = f2.readlines()
        answer_lst = []
        total_f_step = []
        for i, line in tqdm(enumerate(data)):
            line = json.loads(line)
            ans = line['model_answer']
            origin_img_path = json.loads(data2[i])['images'][0]
            #if '\n\n' not in ans:
                #continue
            #try:
            print('------------------------------------')
            score = 0.
            cnt = 0.
            cot_steps = ans.split('\n\n')
            print(cot_steps)
            #print("cot_steps:", cot_steps)
            if len(cot_steps) >= 3:
                cot_steps = cot_steps[1:-1] # GCoT, ReAct: [:-1], other: [1:-1]
                #cot_steps = cot_steps[:-1]
            else:
                cot_steps = cot_steps
            for j, step in enumerate(cot_steps):
                step = re.sub(r'Action:\s*\{[^}]*\};?', '', step)
                #step = re.sub(r'\*\*.*?\*\*', '', step)
                #img = BytesIO(df_dict[i]['image']['bytes'])
                try:
                    #f_step = eval_one_sentence(step, img, model, processor)
                    f_step, _ = eval_one_sentence(step, origin_img_path, model, processor)
                    print("step: ", j, "faithfulness score: ", f_step)
                    score += f_step
                    cnt += 1
                except:
                    continue

            if cnt == 0:
                continue    
            print("total_score:", score / cnt)
            print('------------------------------------')
            total_f_step.append(score / cnt)
            #total_cnt += 1
        
        total_f_step = np.array(total_f_step)
        print("AVG F-step in a Dataset:", np.mean(total_f_step))
        print("STD F-step in a Dataset:", np.std(total_f_step))
            #except:
                #pass
    elif dataset_name == "scienceqa":
        with open('/path/to/cot-faith/results/scienceqa-answer-with-faithact-qwen-alpha-0.6-th-l0p4-h0p6.jsonl', 'r') as f:
            data = f.readlines()
        with open('/path/to/cot-faith/mmhal/mmhal_bench.jsonl', 'r') as f2:
            data2 = f2.readlines()
        answer_lst = []
        total_f_step = []
        for i, line in tqdm(enumerate(data)):
            line = json.loads(line)
            ans = line['model_answer']
            origin_img_path = json.loads(data2[i])['images'][0]
            #if '\n\n' not in ans:
                #continue
            #try:
            print('------------------------------------')
            score = 0.
            cnt = 0.
            cot_steps = ans.split('\n\n')
            print(cot_steps)
            #print("cot_steps:", cot_steps)
            if len(cot_steps) >= 3:
                cot_steps = cot_steps[1:-1] # GCoT, ReAct: [:-1], other: [1:-1]
                #cot_steps = cot_steps[:-1]
            else:
                cot_steps = cot_steps
            for j, step in enumerate(cot_steps):
                step = re.sub(r'Action:\s*\{[^}]*\};?', '', step)
                #step = re.sub(r'\*\*.*?\*\*', '', step)
                #img = BytesIO(df_dict[i]['image']['bytes'])
                try:
                    #f_step = eval_one_sentence(step, img, model, processor)
                    f_step, _ = eval_one_sentence(step, origin_img_path, model, processor)
                    print("step: ", j, "faithfulness score: ", f_step)
                    score += f_step
                    cnt += 1
                except:
                    continue

            if cnt == 0:
                continue    
            print("total_score:", score / cnt)
            print('------------------------------------')
            total_f_step.append(score / cnt)
            #total_cnt += 1
        
        total_f_step = np.array(total_f_step)
        print("AVG F-step in a Dataset:", np.mean(total_f_step))
        print("STD F-step in a Dataset:", np.std(total_f_step))
    """
    if dataset_name == "llava-bench":
        with open('./llava-bench/data.jsonl', 'r') as f:
            data = f.readlines()
        answer_lst = []
        for line in tqdm(data):
            line = json.loads(line)
            query = line['question']
            img = line['images'][0]

            #query = query + ' Think step by step. Your answer should be in the format:\n\n1. xxxxxx. 2.xxxxxx. ...The final answer is xxx.'
            query = query + 'Think step by step.'
            output_text = vlm_agent(query, img, model, processor)
            answer_lst.append({'id': line['idx'], 'model_answer': output_text[0], 'gt': line['answer']})
        
        with open('./llava-bench-answer.jsonl', 'w') as writer:
            for item in answer_lst:
                writer.write(json.dumps(item, ensure_ascii=False)+'\n')

        writer.close()
    """      
