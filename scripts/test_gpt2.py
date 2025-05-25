# chatbot_rag/scripts/test_gpt2.py

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
print("Python executable being used:", sys.executable)

# Load GPT-2 and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

# Decode and print
print("Generated Text:\n", tokenizer.decode(outputs[0]))
