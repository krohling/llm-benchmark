import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

CONTEXT_FILE = 'context.txt'
CONTEXT_TAG = '###CONTEXT###'
PROMPT_TEMPLATE = f'### User:\nUsing the following information, what was the top scientific breakthrough in 2022?\n{CONTEXT_TAG}\n\n### Assistant:\n'
CONTEXT_SIZES = [100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]

tokenizer = AutoTokenizer.from_pretrained("upstage/Llama-2-70b-instruct-v2")

# 8-bit configuration
model = AutoModelForCausalLM.from_pretrained(
    "upstage/Llama-2-70b-instruct-v2",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
)

# 4-bit configuration
# model = AutoModelForCausalLM.from_pretrained(
#     "upstage/Llama-2-70b-instruct-v2",
#     device_map="auto",
#     torch_dtype=torch.float16,
#     load_in_4bit=True,
#     quantization_config=BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type='nf4'
#     ),
#     rope_scaling={"type": "dynamic", "factor": 2} # allows handling of longer inputs
# )

def count_tokens(input):
	tokens = tokenizer(input, return_tensors="pt")
	return len(tokens['input_ids'][0])

with open(CONTEXT_FILE) as f:
    context = f.read()
context_tokens = tokenizer(context, return_tensors="pt")['input_ids'][0]
template_token_length = count_tokens(PROMPT_TEMPLATE.replace(CONTEXT_TAG, ''))

def build_prompt(length):
	token_count = length - template_token_length
	tokens = context_tokens[:token_count]
	text = tokenizer.decode(tokens, skip_special_tokens=True)
	return PROMPT_TEMPLATE.replace(CONTEXT_TAG, text)




for s in CONTEXT_SIZES:
    prompt = build_prompt(s)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Starting generation for size: {s}")
    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=100)
    output_token_count = len(output[0]) - s
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"execution time: {execution_time}s tokens: {output_token_count} t/s: {output_token_count/execution_time}")