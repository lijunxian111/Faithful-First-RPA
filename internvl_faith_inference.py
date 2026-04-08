import math
import numpy as np
import torch
import torchvision.transforms as T
#from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from tqdm import tqdm
import math
import json
from qwen_inference import build_vlm_agent, vlm_agent, qwen_chat_raw

from qwen_eot_inference import ground, poll
from template import *
from PIL import Image
from io import BytesIO
import json
import re

def remove_star_sections(text):
    return re.sub(r'\*\*.*?\*\*', '', text)

import pandas as pd
from world_names import extract_real_objects

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name, model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
path = '/path/to/InternVL3-8B'
device_map = split_model('InternVL3-8B', '/path/to/InternVL3-8B')
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

def bytes_to_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return image


ds_name = "realworldqa"  # "mmhal"  #"realworldqa"
generation_config = dict(max_new_tokens=400, do_sample=False)
helper_model, processor = build_vlm_agent('/path/to/Qwen-2.5-VL')
# set the max number of tiles in `max_num`
if ds_name == "llava-bench":
    with open('./llava-bench/data.jsonl', 'r') as f:
        data = f.readlines()
    with open('/path/to/cot-faith/results/llava-bench-answer-internvl.jsonl', 'r') as f2:
        original_answer = f2.readlines()
    answer_lst = []
    cnt = 0
    for line, ans_line in tqdm(zip(data, original_answer)):
        line = json.loads(line)
        ans_line = json.loads(ans_line)
        query = line['question']
        query = '<image>\n' + query
        img = line['images'][0]

        pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
        org_ans = ans_line['model_answer']
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
                    step = remove_star_sections(step)
                    arr = eval(qwen_chat_raw(TEMPLATE_NOUNS.format(step), helper_model, processor)[0])
                    obj_to_select.extend(arr)
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
            img_size = Image.open(img).size
            if prob > 0.6:
                obj_desc += f'Object exists with probability {prob}. '
                for j in range(len(scores)):
                    bboxes[j] = [bboxes[j][0]/img_size[0]*1000, bboxes[j][1]/img_size[1]*1000, bboxes[j][2]/img_size[0]*1000, bboxes[j][3]/img_size[1]*1000]
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
                obj_desc += '\n'
            elif prob < 0.4:
                continue
                #obj_desc += f'Object doesn\'t exist with probability {1-prob}. '
            else:
                obj_desc += f'Object may exist with probability {prob}. '
                for j in range(len(scores)):
                    bboxes[j] = [bboxes[j][0]/img_size[0]*1000, bboxes[j][1]/img_size[1]*1000, bboxes[j][2]/img_size[0]*1000, bboxes[j][3]/img_size[1]*1000]
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
            #obj_desc += f'Object existing probability: {prob}.'
            region_bank.append(obj_desc)
        
        
        #print(region_bank)
        additional_information = '.'.join(region_bank)
        if additional_information == '':
            final_query = query
        else:
            final_query = TEMPLATE_EOT.format(query, additional_information)
        if cnt == 0:
            print("Final query:", final_query)
        cnt += 1
        #query = query + ' Think step by step. Your answer should be in the format:\n\n1. xxxxxx. 2.xxxxxx. ...The final answer is xxx.'
        #query = query + 'Think step by step. Your reasoning steps should be seperated by \n\n.'
        #output_text = vlm_agent(query, img, model, processor)
        response = model.chat(tokenizer, pixel_values, final_query, generation_config)
        print(f'Assistant: {response}')
        answer_lst.append({'id': line['idx'], 'model_answer': response, 'gt': line['answer']})
            
    with open('./results/llava-bench-answer-internvl-withfaith.jsonl', 'w') as writer:
        for item in answer_lst:
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')

        writer.close()

if ds_name == "mmhal":
    with open('/path/to/cot-faith/mmhal/mmhal_bench.jsonl', 'r') as f:
        data = f.readlines()
    with open('/path/to/cot-faith/mmhal-bench-answer-internvl.jsonl', 'r') as f2:
        original_answer = f2.readlines()
    answer_lst = []
    cnt = 0
    for line, ans_line in tqdm(zip(data, original_answer)):
        line = json.loads(line)
        ans_line = json.loads(ans_line)
        query = line['question']
        query = '<image>\n' + query
        img = line['images'][0]

        pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
        org_ans = ans_line['model_answer']
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
                    step = remove_star_sections(step)
                    e = eval(qwen_chat_raw(TEMPLATE_NOUNS.format(step), helper_model, processor)[0])
                    obj_to_select.extend(e)
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
            img_size = Image.open(img).size
            if prob > 0.6:
                obj_desc += f'Object exists with probability {prob}. '
                for j in range(len(scores)):
                    bboxes[j] = [bboxes[j][0]/img_size[0]*1000, bboxes[j][1]/img_size[1]*1000, bboxes[j][2]/img_size[0]*1000, bboxes[j][3]/img_size[1]*1000]
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
                obj_desc += '\n'
            elif prob < 0.4:
                continue
                #obj_desc += f'Object doesn\'t exist with probability {1-prob}. '
            else:
                obj_desc += f'Object may exist with probability {prob}. '
                for j in range(len(scores)):
                    bboxes[j] = [bboxes[j][0]/img_size[0]*1000, bboxes[j][1]/img_size[1]*1000, bboxes[j][2]/img_size[0]*1000, bboxes[j][3]/img_size[1]*1000]
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
            #obj_desc += f'Object existing probability: {prob}.'
            region_bank.append(obj_desc)
        
        
        #print(region_bank)
        additional_information = '.'.join(region_bank)
        if additional_information == '':
            final_query = query
        else:
            final_query = TEMPLATE_EOT.format(query, additional_information)
        if cnt == 0:
            print("Final query:", final_query)
        cnt += 1
        #query = query + ' Think step by step. Your answer should be in the format:\n\n1. xxxxxx. 2.xxxxxx. ...The final answer is xxx.'
        #query = query + 'Think step by step. Your reasoning steps should be seperated by \n\n.'
        #output_text = vlm_agent(query, img, model, processor)
        response = model.chat(tokenizer, pixel_values, final_query, generation_config)
        print(f'Assistant: {response}')
        answer_lst.append({'id': line['idx'], 'model_answer': response, 'gt': line['answer']})
            
    with open('./results/mmhal-answer-internvl-withfaith.jsonl', 'w') as writer:
        for item in answer_lst:
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')

        writer.close()

elif ds_name == "pope":
    with open('/path/to/cot-faith/train_pope/eval_data_pope.jsonl', 'r') as f:
        data = f.readlines()
    with open('/path/to/cot-faith/pope-answer-internvl.jsonl', 'r') as f2:
        original_answer = f2.readlines()
    with open('/path/to/cot-faith/train_pope/idxes.txt', 'r') as f3:
        idxes = f3.readlines()
    idxes = [int(x.strip()) for x in idxes]
    data = [data[i] for i in idxes]
    original_answer = [original_answer[i] for i in idxes]
    answer_lst = []
    cnt = 0
    for line, ans_line in tqdm(zip(data, original_answer)):
        line = json.loads(line)
        ans_line = json.loads(ans_line)
        if ans_line['id'] >= 7950:
            continue
        query = line['question']
        query = '<image>\n' + query
        img = line['image']

        pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
        org_ans = ans_line['model_answer']
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
                    step = remove_star_sections(step)
                    e = eval(qwen_chat_raw(TEMPLATE_NOUNS.format(step), helper_model, processor)[0])
                    obj_to_select.extend(e)
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
            img_size = Image.open(img).size
            if prob > 0.6:
                obj_desc += f'Object exists with probability {prob}. '
                for j in range(len(scores)):
                    bboxes[j] = [bboxes[j][0]/img_size[0]*1000, bboxes[j][1]/img_size[1]*1000, bboxes[j][2]/img_size[0]*1000, bboxes[j][3]/img_size[1]*1000]
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
                obj_desc += '\n'
            elif prob < 0.4:
                continue
                #obj_desc += f'Object doesn\'t exist with probability {1-prob}. '
            else:
                obj_desc += f'Object may exist with probability {prob}. '
                for j in range(len(scores)):
                    bboxes[j] = [bboxes[j][0]/img_size[0]*1000, bboxes[j][1]/img_size[1]*1000, bboxes[j][2]/img_size[0]*1000, bboxes[j][3]/img_size[1]*1000]
                    obj_desc += f'{j+1}.bbox: {bboxes[j]} '
            #obj_desc += f'Object existing probability: {prob}.'
            region_bank.append(obj_desc)
        
        
        #print(region_bank)
        additional_information = '.'.join(region_bank)
        if additional_information == '':
            final_query = query
        else:
            final_query = TEMPLATE_EOT.format(query, additional_information)
        if cnt == 0:
            print("Final query:", final_query)
        cnt += 1
        #query = query + ' Think step by step. Your answer should be in the format:\n\n1. xxxxxx. 2.xxxxxx. ...The final answer is xxx.'
        #query = query + 'Think step by step. Your reasoning steps should be seperated by \n\n.'
        #output_text = vlm_agent(query, img, model, processor)
        response = model.chat(tokenizer, pixel_values, final_query, generation_config)
        print(f'Assistant: {response}')
        answer_lst.append({'id': ans_line['id'], 'question': line['question'], 'model_answer': response, 'gt': line['answer']})
            
    with open('./results/pope-answer-internvl-withfaith.jsonl', 'w') as writer:
        for item in answer_lst:
            writer.write(json.dumps(item, ensure_ascii=False)+'\n')

        writer.close()
    
elif ds_name == "realworldqa":
        answer_lst = []
        df = pd.read_parquet('/path/to/cot-faith/realworldqa/data/test-00000-of-00002.parquet')
        df2 = pd.read_parquet('/path/to/cot-faith/realworldqa/data/test-00001-of-00002.parquet')
        df = pd.concat([df, df2], ignore_index=True)
        df_dict = df.to_dict(orient='records')
        print("Total samples:", len(df_dict))
        cnt = 0
        with open('/path/to/cot-faith/results/realworldqa-answer-internvl.jsonl', 'r') as f:
            data = f.readlines()
        with open('/path/to/cot-faith/results/realworldqa-answer-internvl-2.jsonl', 'r') as f2:
            data2 = f2.readlines()
        original_answer = data + data2
        for idx, (item, ans_line) in enumerate(zip(df_dict, original_answer)):
            img = BytesIO(item['image']['bytes'])
            if 'idx' not in item:
                item['idx'] = idx
            query = item['question']
            gt = item['answer']
            ans_line = json.loads(ans_line)
            org_ans = ans_line['model_answer']

            query = query.replace('Please answer directly with only the letter of the correct option and nothing else.', '')
            #query = query + 'Think step by step.'
            pixel_values = load_image(img, max_num=12).to(torch.bfloat16).cuda()
            org_ans = ans_line['model_answer']
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
                        step = remove_star_sections(step)
                        #e = extract_real_objects(step)
                        e = eval(qwen_chat_raw(TEMPLATE_NOUNS.format(step), helper_model, processor)[0])
                        obj_to_select.extend(e)
                    except:
                        continue
                obj_to_ground = list(set(obj_to_select))

            region_bank = []
            prob_lst = []
        
            for obj in obj_to_ground:
                scores, bboxes = ground(img, obj)
                obj_desc = f"{obj}: "
                prob_dict = poll(img, scores, obj, no_poll=True)
                #print(prob_dict)
                prob_lst.append(prob_dict)
                #for j in range(len(scores)):
                    #obj_desc += f'{j+1}.bbox: {bboxes[j]}; confidence: {scores[j]} '
                prob = prob_dict['p_yes'].item()
                img_size = Image.open(img).size
                if prob > 0.6:
                    obj_desc += f'Object exists with probability {prob}. '
                     #no ground
                    for j in range(len(scores)):
                        bboxes[j] = [math.ceil(bboxes[j][0]/img_size[0]*1000), math.ceil(bboxes[j][1]/img_size[1]*1000), math.ceil(bboxes[j][2]/img_size[0]*1000), math.ceil(bboxes[j][3]/img_size[1]*1000)]
                        obj_desc += f'{j+1}.bbox: {bboxes[j]} '
                    
                    obj_desc += '\n'
                elif prob < 0.4:
                    continue
                    #obj_desc += f'Object doesn\'t exist with probability {1-prob}. '
                else:
                    obj_desc += f'Object may exist with probability {prob}. '
                    #no ground
                    for j in range(len(scores)):
                        bboxes[j] = [math.ceil(bboxes[j][0]/img_size[0]*1000), math.ceil(bboxes[j][1]/img_size[1]*1000), math.ceil(bboxes[j][2]/img_size[0]*1000), math.ceil(bboxes[j][3]/img_size[1]*1000)]
                        obj_desc += f'{j+1}.bbox: {bboxes[j]} '
                    
            #obj_desc += f'Object existing probability: {prob}.'
                region_bank.append(obj_desc)
        
        
        #print(region_bank)
            additional_information = '\n'.join(region_bank)
            if additional_information == '':
                final_query = query
            else:
                final_query = TEMPLATE_EOT.format(query, additional_information)
            if cnt == 0:
                print("Final query:", final_query)
            cnt += 1
        #query = query + ' Think step by step. Your answer should be in the format:\n\n1. xxxxxx. 2.xxxxxx. ...The final answer is xxx.'
        #query = query + 'Think step by step. Your reasoning steps should be seperated by \n\n.'
        #output_text = vlm_agent(query, img, model, processor)
            response = model.chat(tokenizer, pixel_values, final_query, generation_config)
            print(f'Assistant: {response}')
            answer_lst.append({'id': item['idx'], 'model_answer': response, 'gt': item['answer']})

            #query = query + ' Think step by step. Your answer should be in the format:\n\n1. xxxxxx. 2.xxxxxx. ...The final answer is xxx.'
            #query = query + 'Think step by step. Your reasoning steps should be seperated by \n\n.'
            #output_text = vlm_agent(query, img, model, processor)
            #response = model.chat(tokenizer, pixel_values, query, generation_config)
            #print(f'[{idx}] User: {query}\nAssistant: {response}')
            #answer_lst.append({'id': idx, 'model_answer': response, 'gt': gt})

        with open('./realworldqa-answer-internvl-with-faith-no-poll.jsonl', 'w') as writer:
            for item in answer_lst:
                writer.write(json.dumps(item, ensure_ascii=False)+'\n')
        writer.close()
"""
pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
generation_config = dict(max_new_tokens=1024, do_sample=True)


question = '<image>\nPlease describe the image shortly.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')
"""
