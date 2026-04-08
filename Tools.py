from pathlib import Path
from typing import Tuple, Union
import inspect
import json

import torch
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, CLIPProcessor

from template import TEMPLATE_REACT_AGENT_INTRO, TEMPLATE_REACT_INITIAL_USER
from train_pope.model import CLIP_cls
import numpy as np
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
BEST_CKPT_PATH = BASE_DIR  / "path" / "to" / "cot-faith" / "save_pope_clip" / "best.pt"
GROUNDING_DINO_REPO = "IDEA-Research/grounding-dino-base"
CLIP_MODEL_REPO = "openai/clip-vit-large-patch14-336"
_GROUNDING_POST_PROCESS_PARAMS = None
_GROUNDING_COMPONENTS = None
_CLIP_COMPONENTS = None

REACT_ALLOWED_TOOLS = {"inspect_object", "finish"}
REACT_INABILITY_KEYWORDS = ("cannot", "can't", "unable", "without", "no ")
REACT_IMAGE_TERMS = ("image", "photo", "picture", "visual", "sight")


def _load_image(image_source: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")
    if isinstance(image_source, (str, Path)):
        return Image.open(image_source).convert("RGB")
    raise TypeError(f"Unsupported image source type: {type(image_source)!r}")


def _post_process_grounding(processor, outputs, input_ids, target_sizes):
    global _GROUNDING_POST_PROCESS_PARAMS
    if _GROUNDING_POST_PROCESS_PARAMS is None:
        sig = inspect.signature(processor.post_process_grounded_object_detection)
        _GROUNDING_POST_PROCESS_PARAMS = tuple(sig.parameters.keys())

    kwargs = {}
    params = _GROUNDING_POST_PROCESS_PARAMS
    if "input_ids" in params:
        kwargs["input_ids"] = input_ids
    if "target_sizes" in params:
        kwargs["target_sizes"] = target_sizes
    if "box_threshold" in params:
        kwargs["box_threshold"] = 0.35
    if "text_threshold" in params:
        kwargs["text_threshold"] = 0.25
    # Some releases expect a single threshold argument.
    if "threshold" in params and "box_threshold" not in params:
        kwargs["threshold"] = 0.35

    return processor.post_process_grounded_object_detection(outputs, **kwargs)


def _get_grounding_components():
    global _GROUNDING_COMPONENTS
    if _GROUNDING_COMPONENTS is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(GROUNDING_DINO_REPO, trust_remote_code=True)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            GROUNDING_DINO_REPO, trust_remote_code=True
        ).to(device)
        model.eval()
        _GROUNDING_COMPONENTS = (processor, model, device)
    return _GROUNDING_COMPONENTS


def ground(image_path, span: str, k=5):
    processor, model, device = _get_grounding_components()

    #image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    #image = Image.open(requests.get(image_url, stream=True).raw)
    # Check for cats and remote controls
    # VERY important: text queries need to be lowercased + end with a dot
    image = _load_image(image_path)
    span = span + '.'
    inputs = processor(images=image, text=span, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = _post_process_grounding(
        processor,
        outputs,
        input_ids=inputs.input_ids,
        target_sizes=[image.size[::-1]],
    )
    #print(results)
    results = results[0]
    #print(results.keys())
    scores = results['scores'].detach().cpu().numpy().tolist()
    bboxes = results['boxes'].detach().cpu().numpy().tolist()
    return scores, bboxes

def poll(image_path, scores_from_ground, span, alpha=0.7):
    p_yes_candidate = max(scores_from_ground) if len(scores_from_ground)>0 else 0.0
    processor, clip, device = _get_clip_components()
    inputs = processor(
        text=[span],
        images=_load_image(image_path),
        return_tensors="pt",
        padding=True,
    ).to(device)
    #print(inputs.keys())
    output_logits = clip(**inputs)
    #print(outputs.keys())
    #logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    #probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
    probs = output_logits
    sim = probs[:,1].detach().cpu()
    """
    text = clip.tokenize([span]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        sim = (image_features @ text_features.T).item()
    """
    p_yes_clip = torch.sigmoid(sim)
    #print(p_yes_clip)
    
    p_yes = alpha*p_yes_clip + (1-alpha)*p_yes_candidate
    p_no = 1 - p_yes
    
    return {"p_yes": p_yes, "p_no": p_no}


def _get_clip_components():
    global _CLIP_COMPONENTS
    if _CLIP_COMPONENTS is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_REPO, trust_remote_code=True)
        clip = CLIP_cls(CLIP_MODEL_REPO).to(device)
        state_dict = torch.load(BEST_CKPT_PATH, map_location=device)
        clip.load_state_dict(state_dict)
        clip.eval()
        _CLIP_COMPONENTS = (processor, clip, device)
    return _CLIP_COMPONENTS


def preload_tool_models():
    _get_grounding_components()
    _get_clip_components()


def inspect_object(image, object_name: str) -> str:
    obj = (object_name or "").strip()
    if not obj:
        return "inspect_object received an empty object name."

    try:
        scores, bboxes = ground(image, obj)
    except Exception as exc:
        return f"Unable to ground '{obj}': {exc}"

    try:
        prob_dict = poll(image, scores, obj)
        prob = prob_dict["p_yes"]
        confidence = float(prob.item() if hasattr(prob, "item") else prob)
    except Exception as exc:
        return f"Polling failed for '{obj}': {exc}"

    desc = f"{obj}: existence probability {confidence:.2f}. "
    if confidence > 0.6:
        desc += "Likely present. "
    elif confidence < 0.4:
        desc += "Likely absent. "
    else:
        desc += "Uncertain presence. "

    if scores and bboxes:
        bbox_descriptions = []
        for idx, (bbox, score) in enumerate(zip(bboxes, scores), start=1):
            rounded_bbox = [round(coord, 2) for coord in bbox]
            bbox_descriptions.append(
                f"{idx}. bbox: {rounded_bbox}, confidence: {score:.2f}"
            )
        desc += " ".join(bbox_descriptions)

    return desc


def is_inability_response(text: str) -> bool:
    lowered = (text or "").strip().lower()
    if not lowered:
        return False
    if any(keyword in lowered for keyword in REACT_INABILITY_KEYWORDS) and any(
        term in lowered for term in REACT_IMAGE_TERMS
    ):
        return True
    return False


def parse_react_action(agent_output: str) -> Tuple[str, str]:
    action_line = None
    for line in reversed(agent_output.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("Action:"):
            action_line = stripped[len("Action:"):].strip()
            break
    if action_line is None:
        raise ValueError("Missing Action line.")
    try:
        payload = json.loads(action_line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid action JSON: {exc}") from exc

    tool = payload.get("tool")
    tool_input = payload.get("input")
    if tool is None:
        raise ValueError("Action JSON must include a 'tool' field.")
    if tool_input is None:
        tool_input = ""
    elif isinstance(tool_input, (list, dict)):
        raise ValueError("Action input must be provided as a plain string.")
    else:
        tool_input = str(tool_input)
    return tool, tool_input


def build_react_initial_user_text(question: str) -> str:
    return TEMPLATE_REACT_INITIAL_USER.format(question=question)


def build_react_bootstrap_messages(question, image):
    initial_user_text = build_react_initial_user_text(question)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": TEMPLATE_REACT_AGENT_INTRO}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": initial_user_text},
            ],
        },
    ]
    return messages


def make_observation_message(observation: str) -> dict:
    return {"role": "user", "content": [{"type": "text", "text": observation}]}
