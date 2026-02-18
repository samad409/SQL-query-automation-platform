import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

print("Loading trained model...")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("model")
print("Model ready.")

while True:
    question = input("Enter question: ")
    input_text = "translate English to SQL: " + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=100)
    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated SQL:", sql)