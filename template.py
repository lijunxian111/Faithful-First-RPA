import os


TEMPLATE_REGIONS = """Based on the user question\n\n###{}###\n\n and the image, 
think about any object in the image helping you answer this question. Only find objects that refer to tangible things, not abstract concepts, actions, Proper Nouns (people name, place name, organization or title) or locations. Do not include any non-object nouns or words like 'Image' or 'Photo'. Your answer should be
only a list of object names and no other words. Like: ['xxx','xxx','xxx']. 'xxx' means an object name.
"""

TEMPLATE_REGIONS_TEXT = """Based on the user question\n\n###{}###\n\n, 
think about any object helping you answer this question. Only find objects that refer to tangible things, not abstract concepts, actions, Proper Nouns (people name, place name, organization or title) or locations. Do not include any non-object words or words like 'Image' or 'Photo'. Your answer should be
only a list of object names and no other words. Like: ['xxx','xxx','xxx']. 'xxx' means an object name.
"""

TEMPLATE_EOT = """{}.\n\nAdditional location information:\n\n{}\n\nUse only the objects mentioned in additional information. Note that objects we list here surely occur in the image. Do not include new objects or descriptions. Do not repeat the evidences, confidence scores and bounding boxes in your reasoning. Think step by step. Steps should be like: 1.<object>:<analysis>\n\n 2.<object>:<analysis>\n\n...\n\nThus, <final answer related to the question>."""

TEMPLATE_NOUNS = """Extract all objects mentioned in the following sentence (even if it says no xxx). Only extract nouns meaning objects, not abstract adjectives, concepts, actions, general nouns or locations. Do not include non-object nouns or words like 'Image', 'Object', 'Feature', or 'Photo'.\n\n###{}###\n\nReturn only a list of nouns like ['xxx', 'xxx', 'xxx'] and do not include any other things."""

TEMPLATE_EOT_REFINE = """
Task: Revise the reasoning so it is fully consistent with the verified evidence. Use objects mentioned, remove unsupported objects. 

Input:
Image: The given image.
Question: 
{}

Evidences of Objects: {}

Note that you shouldn't repeat the evidences, confidence scores and bounding boxes in your reasoning. Obey the evidence but not your belief. Answer should be in the format:
Revised Reasoning Steps:\n\n<your reasoning steps only>\n\n<Final Answer: your final answer>.
"""

TEMPLATE_GCOT =  """
You are a visual reasoning assistant.
Follow this process strictly:

Step 1. Grounding: Identify and describe the region(s) in the image relevant to the question.
Step 2. Reasoning: Use the grounded evidence to reason step by step.
Step 3. Answer: Provide your final concise answer.

Question: {}
"""
