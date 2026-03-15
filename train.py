import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments

MODEL_NAME = "t5-small"

print("Loading dataset...")

df = pd.read_csv("dataset/text_to_sql_dataset_5000.csv")

# Optional: use only part of dataset for faster training
# df = df.sample(3000)

train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

print("Loading tokenizer and model...")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


def preprocess(example):

    input_text = "translate English to SQL: " + example["question"]
    target_text = example["sql"]

    inputs = tokenizer(
        input_text,
        max_length=64,
        padding="max_length",
        truncation=True
    )

    labels = tokenizer(
        target_text,
        max_length=64,
        padding="max_length",
        truncation=True
    )

    inputs["labels"] = labels["input_ids"]

    return inputs


print("Tokenizing dataset...")

train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

train_dataset.set_format(type="torch")
val_dataset.set_format(type="torch")

training_args = TrainingArguments(
    output_dir="results",
    num_train_epochs=2,                 # reduced epochs
    per_device_train_batch_size=16,     # bigger batch
    logging_steps=50,
    save_steps=500,
    fp16=torch.cuda.is_available(),     # faster on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

print("Training started...")

trainer.train()

print("Saving model...")

model.save_pretrained("model")
tokenizer.save_pretrained("model")

print("Training complete!")