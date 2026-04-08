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

TEMPLATE_NOUNS = """Extract all objects mentioned in the following sentence (even if it says no xxx). Only extract nouns meaning objects, not abstract adjectives, concepts, actions, general nouns or locations. Do not include non-object nouns or words like 'Image', 'Object', 'Feature', or 'Photo'.\n\n###{}###\n\nReturn only a list of single-word nouns like ['xxx', 'xxx', 'xxx'] and do not include any other things."""

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


TEMPLATE_REACT_FINAL = """You are solving the following visual question using the ReAct method.

Question:
{}

Here is the transcript of your tool-assisted reasoning:
{}

Using the observations, provide a final response. Even if the transcript is empty or inconclusive, directly analyse the image with your native vision-language ability and give your best supported answer. Do not respond with an inability statement; instead make a well-reasoned attempt grounded in the visual evidence. Output two sections separated by a blank line: start with `Reasoning:` followed by a thorough explanation grounded in the transcript and image, then after a blank line output `Answer:` followed by the concise final answer."""

TEMPLATE_REACT_AGENT_INTRO = """You are a vision-language ReAct agent. Answer the user's question by reasoning about the image and calling tools when needed.

Tools you can call:
- inspect_object: input must be a single object name as a plain string, e.g. "traffic light". Runs detection + CLIP scoring and returns observations about that object.
- finish: input must be your final answer as a plain string. Use this when you are ready to respond to the user.

You must always respond with two lines:
Thought: <your reasoning in natural language>
Action: {{"tool": "<tool name>", "input": <string or null>}}

Only use the tools defined above. Do not invent new tools. Always supply the tool input exactly as specified (never use objects, lists, or additional fields). If you ever mention a tool name that is not in the list, you must immediately correct yourself and choose one of the allowed tools or call finish. Continue thinking step by step until you call finish. If inspect_object is unnecessary or you already know the answer, rely on your direct perception of the image and provide the most plausible and informative answer you can. Avoid saying you cannot identify something; always give your best supported response.

When you call finish, return your final message formatted exactly as:
Reasoning: <comprehensive explanation grounded in the dialogue and image>

Answer: <concise final answer>
"""

TEMPLATE_REACT_AGENT_STEP = """Question:
{question}

History:
{history}

Remember:
- Think before acting.
- When confident, call finish with your final answer.
- If tools are unhelpful, trust your own vision-language understanding to answer directly.
- Never use tool names outside of ['inspect_object', 'finish'].
- Ensure the Action line always uses the exact JSON `{{"tool": "...", "input": "..."}}` with the input formatted as described above.
- Call inspect_object only when you need targeted details about a specific object in the scene.

{instruction}
"""


TEMPLATE_REACT_INITIAL_USER = """Question:
{question}

Respond using two lines:
Thought: <your reasoning in natural language>
Action: {{"tool": "<tool name>", "input": "<string>"}}

Remember:
- Think before acting.
- When confident, call finish with your final answer.
- If tools are unhelpful, trust your own vision-language understanding to answer directly.
- Never use tool names outside of ['inspect_object', 'finish'].
- Ensure the Action line always uses the exact JSON {{"tool": "...", "input": "..."}} with the input formatted as described above.
- Call inspect_object only when you need targeted details about a specific object in the scene."""
