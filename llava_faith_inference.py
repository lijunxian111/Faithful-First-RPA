import os
import json
import torch 
import torch.nn as nn
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM

from PIL import Image
from template import *
# This utility is compatible with LLaVA as well
from qwen_vl_utils import process_vision_info
from transformers import CLIPProcessor
from train_pope.model import CLIP_cls
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from Tools import poll, ground
from template import TEMPLATE_EOT , TEMPLATE_NOUNS
from vat_utils import iterate_mmhal_bench, iterate_pope

# --- LLaVA Model Functions ---

def build_vlm_agent(model_path):
    """
    Loads the LLaVA model and processor.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

def vlm_agent(TEXT, image_path, model, processor):
    """
    Processes a text prompt and an image with the VLM model.
    """
    # The image_path can be a file path or a PIL.Image object
    if isinstance(image_path, str):
        image = Image.open(image_path).convert("RGB")
    else:
        image = image_path

    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": TEXT}]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text

def llm_chat_raw(query, model, processor):
    """
    Processes a text-only prompt with the model.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": query}]}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=None, padding=True, return_tensors="pt")
    inputs = inputs.to(next(model.parameters()).device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return output_text



def load_existing_ids(path):
    existing_ids = set()
    if not os.path.exists(path):
        return existing_ids
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'id' in data:
                existing_ids.add(str(data['id']))
    return existing_ids


def load_base_answers(path):
    answers = {}
    if not os.path.exists(path):
        print(f"Warning: base answer file '{path}' not found. Skipping related items.")
        return answers
    with open(path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'id' in data:
                answers[str(data['id'])] = data
    return answers


if __name__ == "__main__":
    # --- Main Change: Update the model path to the LLaVA model ---
    model_path = "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    model, processor = build_vlm_agent(model_path)
    ds_names = ["pope"]

    for ds_name in ds_names:
        if ds_name == "llava-bench":
            output_filename = "./llava-bench-answer-with-faithact-llava.jsonl"
            base_answer_path = "./llava-bench-answer-llava-onevision.jsonl"
            existing_ids = load_existing_ids(output_filename)
            base_answers = load_base_answers(base_answer_path)

            print("Loading 'LLaVA-Bench-in-the-Wild' dataset from Hugging Face Hub...")
            dataset = load_dataset("lmms-lab/LLaVA-Bench-in-the-Wild", split="train")
            print(f"Dataset loaded. Found {len(existing_ids)} existing results to skip.")

            with open(output_filename, "a") as writer:
                for item in tqdm(dataset, desc="Processing LLaVA-Bench with EOT"):
                    question_id = str(item["question_id"])
                    if question_id in existing_ids:
                        continue

                    base_answer = base_answers.get(question_id)
                    if not base_answer:
                        if base_answers:
                            print(f"Base answer missing for question_id {question_id}. Using direct query.")
                        base_answer = {}

                    query = item["question"]
                    img = item["image"]
                    gt = item.get("caption", "")
                    org_ans = base_answer.get("model_answer", "")

                    if "\n\n" not in org_ans:
                        obj_to_ground = []
                    else:
                        cot_steps = org_ans.split("\n\n")
                        cot_steps = cot_steps[1:-1] if len(cot_steps) > 1 else []

                        obj_to_select = []
                        for step in cot_steps:
                            try:
                                # --- Main Change: Use the new text-only function ---
                                response = llm_chat_raw(TEMPLATE_NOUNS.format(step), model, processor)
                                obj_to_select.extend(eval(response[0]))
                            except Exception:
                                continue
                        obj_to_ground = list(set(obj_to_select))

                    region_bank = []

                    for obj in obj_to_ground:
                        scores, bboxes = ground(img, obj)
                        prob_dict = poll(img, scores, obj)
                        prob = prob_dict["p_yes"].item()

                        obj_desc = f"{obj}: "
                        if prob > 0.6:
                            obj_desc += f"Object exists with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        elif prob < 0.4:
                            obj_desc += f"Object may not exist with probability {1 - prob:.2f}. "
                        else:
                            obj_desc += f"Object may exist with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        region_bank.append(obj_desc)

                    print(f"Region Bank: {region_bank}")
                    additional_information = ". ".join(region_bank)

                    if not additional_information:
                        final_query = query
                    else:
                        final_query = TEMPLATE_EOT.format(query, additional_information)

                    output_text = vlm_agent(final_query, img, model, processor)
                    print(f"[{question_id}] Final Answer: {output_text[0]}")
                    result = {"id": item["question_id"], "model_answer": output_text[0], "gt": gt}
                    writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                    writer.flush()

            print(f"\nProcessing complete. Results saved to {output_filename}")

        elif ds_name == "realworldqa":
            output_filename = "./realworldqa-answer-with-faithact-llava.jsonl"
            base_answer_path = "./realworldqa-answer-llava-onevision.jsonl"
            existing_ids = load_existing_ids(output_filename)
            base_answers = load_base_answers(base_answer_path)

            print("Loading 'RealworldQA' dataset from Hugging Face Hub...")
            dataset = load_dataset("xai-org/RealworldQA", split="test")
            print(f"Dataset loaded. Found {len(existing_ids)} existing results to skip.")

            with open(output_filename, "a") as writer:
                for idx, item in enumerate(tqdm(dataset, desc="Processing RealworldQA with EOT")):
                    idx_key = str(idx)
                    if idx_key in existing_ids:
                        continue

                    base_answer = base_answers.get(idx_key)
                    if not base_answer:
                        if base_answers:
                            print(f"Base answer missing for idx {idx}. Using direct query.")
                        base_answer = {}

                    query = item["question"]

                    query = query.replace("Please answer directly with only the letter of the correct option and nothing else.", "")

                    img = item["image"]
                    gt = item.get("answer", "")
                    org_ans = base_answer.get("model_answer", "")

                    if "\n\n" not in org_ans:
                        obj_to_ground = []
                    else:
                        cot_steps = org_ans.split("\n\n")
                        cot_steps = cot_steps[1:-1] if len(cot_steps) > 1 else []

                        obj_to_select = []
                        for step in cot_steps:
                            try:
                                response = llm_chat_raw(TEMPLATE_NOUNS.format(step), model, processor)
                                obj_to_select.extend(eval(response[0]))
                            except Exception:
                                continue
                        obj_to_ground = list(set(obj_to_select))

                    region_bank = []
                    for obj in obj_to_ground:
                        scores, bboxes = ground(img, obj)
                        prob_dict = poll(img, scores, obj)
                        prob = prob_dict["p_yes"].item()

                        obj_desc = f"{obj}: "
                        if prob > 0.6:
                            obj_desc += f"Object exists with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        elif prob < 0.4:
                            obj_desc += f"Object may not exist with probability {1 - prob:.2f}. "
                        else:
                            obj_desc += f"Object may exist with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        region_bank.append(obj_desc)

                    print(f"[{idx}] Region Bank: {region_bank}")
                    additional_information = ". ".join(region_bank)

                    if not additional_information:
                        final_query = query
                    else:
                        final_query = TEMPLATE_EOT.format(query, additional_information)

                    output_text = vlm_agent(final_query, img, model, processor)
                    print(f"[{idx}] Final Answer: {output_text[0]}")
                    result = {"id": idx, "model_answer": output_text[0], "gt": gt}
                    writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                    writer.flush()

            print(f"\nProcessing complete. Results saved to {output_filename}")

        elif ds_name == "mmhal-bench":
            output_filename = "./mmhal-bench-answer-with-faithact-llava.jsonl"
            base_answer_path = "./mmhal-bench-answer-llava-onevision.jsonl"
            existing_ids = load_existing_ids(output_filename)
            base_answers = load_base_answers(base_answer_path)

            print("Loading local 'MMHal-Bench' dataset dump...")
            samples = iterate_mmhal_bench()
            total = getattr(samples, "total", None)
            print(f"Dataset prepared. Found {len(existing_ids)} existing results to skip.")

            with open(output_filename, "a") as writer:
                for sample in tqdm(samples, desc="Processing MMHal-Bench with EOT", total=total):
                    sample_id = str(sample.sample_id)
                    if sample_id in existing_ids:
                        continue

                    base_answer = base_answers.get(sample_id)
                    if not base_answer:
                        if base_answers:
                            print(f"Base answer missing for sample_id {sample_id}. Using direct query.")
                        base_answer = {}

                    query = sample.question or ""
                    img = sample.image
                    gt = sample.answer
                    org_ans = base_answer.get("model_answer", "")

                    if "\n\n" not in org_ans:
                        obj_to_ground = []
                    else:
                        cot_steps = org_ans.split("\n\n")
                        cot_steps = cot_steps[1:-1] if len(cot_steps) > 1 else []

                        obj_to_select = []
                        for step in cot_steps:
                            try:
                                response = llm_chat_raw(TEMPLATE_NOUNS.format(step), model, processor)
                                obj_to_select.extend(eval(response[0]))
                            except Exception:
                                continue
                        obj_to_ground = list(set(obj_to_select))

                    region_bank = []
                    for obj in obj_to_ground:
                        scores, bboxes = ground(img, obj)
                        prob_dict = poll(img, scores, obj)
                        prob = prob_dict["p_yes"].item()

                        obj_desc = f"{obj}: "
                        if prob > 0.6:
                            obj_desc += f"Object exists with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        elif prob < 0.4:
                            obj_desc += f"Object may not exist with probability {1 - prob:.2f}. "
                        else:
                            obj_desc += f"Object may exist with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        region_bank.append(obj_desc)

                    print(f"[{sample_id}] Region Bank: {region_bank}")
                    additional_information = ". ".join(region_bank)

                    if not additional_information:
                        final_query = query
                    else:
                        final_query = TEMPLATE_EOT.format(query, additional_information)

                    output_text = vlm_agent(final_query, img, model, processor)
                    print(f"[{sample_id}] Final Answer: {output_text[0]}")
                    result = {"id": sample.sample_id, "model_answer": output_text[0], "gt": gt}
                    writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                    writer.flush()

            print(f"\nProcessing complete. Results saved to {output_filename}")

        elif ds_name == "pope":
            output_filename = "./pope-answer-with-faithact-llava.jsonl"
            base_answer_path = "./pope-answer-llava-onevision.jsonl"
            existing_ids = load_existing_ids(output_filename)
            base_answers = load_base_answers(base_answer_path)

            print("Loading local 'POPE' dataset dump...")
            try:
                samples = iterate_pope()
            except FileNotFoundError as exc:
                print(f"POPE data not found, skipping this dataset: {exc}")
                continue
            total = getattr(samples, "total", None)
            print(f"Dataset prepared. Found {len(existing_ids)} existing results to skip.")

            with open(output_filename, "a") as writer:
                for sample in tqdm(samples, desc="Processing POPE with EOT", total=total):
                    sample_id = str(sample.sample_id)
                    if sample_id in existing_ids:
                        continue

                    base_answer = base_answers.get(sample_id)
                    if not base_answer:
                        if base_answers:
                            print(f"Base answer missing for sample_id {sample_id}. Using direct query.")
                        base_answer = {}

                    query = sample.question
                    img = sample.image
                    gt = sample.answer
                    org_ans = base_answer.get("model_answer", "")

                    if "\n\n" not in org_ans:
                        obj_to_ground = []
                    else:
                        cot_steps = org_ans.split("\n\n")
                        cot_steps = cot_steps[1:-1] if len(cot_steps) > 1 else []

                        obj_to_select = []
                        for step in cot_steps:
                            try:
                                response = llm_chat_raw(TEMPLATE_NOUNS.format(step), model, processor)
                                obj_to_select.extend(eval(response[0]))
                            except Exception:
                                continue
                        obj_to_ground = list(set(obj_to_select))

                    region_bank = []
                    for obj in obj_to_ground:
                        scores, bboxes = ground(img, obj)
                        prob_dict = poll(img, scores, obj)
                        prob = prob_dict["p_yes"].item()

                        obj_desc = f"{obj}: "
                        if prob > 0.6:
                            obj_desc += f"Object exists with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        elif prob < 0.4:
                            obj_desc += f"Object may not exist with probability {1 - prob:.2f}. "
                        else:
                            obj_desc += f"Object may exist with probability {prob:.2f}. "
                            for j, (bbox, score) in enumerate(zip(bboxes, scores)):
                                obj_desc += f"{j+1}.bbox: {[round(c, 2) for c in bbox]}; confidence: {score:.2f} "
                        region_bank.append(obj_desc)

                    print(f"[{sample_id}] Region Bank: {region_bank}")
                    additional_information = ". ".join(region_bank)

                    if not additional_information:
                        final_query = query
                    else:
                        final_query = TEMPLATE_EOT.format(query, additional_information)

                    output_text = vlm_agent(final_query, img, model, processor)
                    print(f"[{sample_id}] Final Answer: {output_text[0]}")
                    result = {"id": sample.sample_id, "model_answer": output_text[0], "gt": gt}
                    writer.write(json.dumps(result, ensure_ascii=False) + "\n")
                    writer.flush()

            print(f"\nProcessing complete. Results saved to {output_filename}")
