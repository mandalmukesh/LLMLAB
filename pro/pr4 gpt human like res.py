import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')

prompt = "What is the future of AI?"
inputs = tokenizer_gpt2.encode(prompt, return_tensors="pt")
outputs = model_gpt2.generate(inputs, max_length=50, num_return_sequences=1)
response = tokenizer_gpt2.decode(outputs[0], skip_special_tokens=True)
print("GPT-2 Response:", response)