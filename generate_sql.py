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

history = []

while True:
    question = input("Enter question: ").strip()
    if not question:
        continue
    if question.lower() == "quit":
        break
    if question.lower() == "help":
        print("Commands: quit, help, history")
        continue
    if question.lower() == "history":
        for i, q in enumerate(history, 1):
            print(f"{i}. {q}")
        continue
    history.append(question)
    input_text = PREFIX + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    outputs = model.generate(
        input_ids, max_length=200, num_beams=4,
        early_stopping=True, repetition_penalty=1.2, length_penalty=1.0
    )
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated SQL:", sql)