# SQL Query Automation Platform

This project trains a T5 model to convert English questions into SQL queries, then provides an interactive script to generate SQL from user input.

## Overview

The workflow has two main phases:

1. Training (`train.py`)
2. Inference (`generate_sql.py`)

Training uses `dataset/text_to_sql_dataset_5000.csv` with two columns:

- `question`: natural language text
- `sql`: target SQL query

The model is fine-tuned from `t5-small` and saved locally in the `model/` folder.

## Project Structure

```
SQL-query-automation-platfrom/
├── dataset/
│   └── text_to_sql_dataset_5000.csv
├── model/                         # saved trained model + tokenizer
├── results/                       # trainer output and checkpoints
├── train.py                       # fine-tuning script
├── generate_sql.py                # interactive SQL generation script
├── requirements.txt
└── README.md
```

## How It Works

### 1. Training Pipeline (`train.py`)

`train.py` performs the following steps:

1. Loads dataset using Pandas.
2. Splits data into train/validation/test using `train_test_split`.
3. Converts Pandas DataFrames to Hugging Face `Dataset` objects.
4. Preprocesses each sample with a prompt prefix:

```text
translate English to SQL: <question>
```

5. Tokenizes both input question and target SQL (`max_length=64`, padded and truncated).
6. Fine-tunes `T5ForConditionalGeneration` with Hugging Face `Trainer`.
7. Saves the fine-tuned model and tokenizer to `model/`.

Training output checkpoints are stored in `results/` (for example: `results/checkpoint-500`).

### 2. Inference Pipeline (`generate_sql.py`)

`generate_sql.py`:

1. Loads tokenizer (`t5-small`) and trained model from `model/`.
2. Detects device automatically (`cuda` if available, else `cpu`).
3. Runs an interactive CLI loop for user questions.
4. Builds model input using the same prefix used in training.
5. Generates SQL with beam search settings:
   - `max_length=200`
   - `num_beams=4`
   - `early_stopping=True`
   - `repetition_penalty=1.2`
   - `length_penalty=1.0`
6. Prints generated SQL and generation time.

Supported CLI commands:

- `help`: show commands
- `history`: show asked questions
- `clear`: clear question history
- `quit`: exit program

## Installation

1. Create and activate a virtual environment.

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Initialize the local database

```bash
python build_database.py
```

Run this once before using the interactive SQL generator so tables and sample rows are available.

### Train the model

```bash
python train.py
```

After training, `model/` will contain the saved weights and tokenizer files.

### Generate SQL interactively

```bash
python generate_sql.py
```

Example:

```text
Enter question: show all customers from california
Generated SQL: SELECT * FROM customers WHERE state = 'california';
```

## Dependencies

From `requirements.txt`:

- torch
- transformers
- datasets
- pandas
- scikit-learn
- tabulate

## Notes

- If GPU is available, training/inference can run faster with CUDA.
- Keep training prefix format consistent (`translate English to SQL:`) across training and inference for best results.