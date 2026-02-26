import torch
import logging
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration

VERSION = "1.0.0"
MODEL_PATH = "model"
TOKENIZER_NAME = "t5-small"
PREFIX = "translate English to SQL: "

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading trained model...")
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to(device)
print("Model ready.")

while True:
    question = input("Enter question: ")
    input_text = "translate English to SQL: " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(input_ids, max_length=100)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated SQL:", sql)