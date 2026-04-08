from transformers import AutoTokenizer
import json

def token_length_stats_with_tokenizer(string_list, model_name='/path/to/Qwen-2.5-VL'):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    token_lengths = []
    for text in string_list:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > 10:
            token_lengths.append(len(tokens))
    
    avg_length = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    max_length = max(token_lengths) if token_lengths else 0
    min_length = min(token_lengths) if token_lengths else 0
    
    return avg_length, max_length, min_length

path = "/path/to/realworldqa-answer-qwen.jsonl"
with open(path, 'r') as f:
    data = f.readlines()
string_list = [json.loads(line)['model_answer'] for line in data]
#string_list = ["This is a test sentence.", "Another example sentence.", "Short one."]
avg, max_len, min_len = token_length_stats_with_tokenizer(string_list)

print(f"Average token length: {avg}")
print(f"Longest token length: {max_len}")
print(f"Shortest token length: {min_len}")
