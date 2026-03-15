# SQL Query Automation Platform

An end-to-end Text-to-SQL project that fine-tunes a T5 model to convert natural-language questions into SQL, then executes generated SQL against a local SQLite database for immediate results.

This repository contains three main runtime components:

- A training pipeline to fine-tune T5 on question-to-SQL pairs.
- An inference CLI that translates user questions into SQL.
- A local SQLite bootstrap script that creates schema and seed data for testing.

## 1. Project Goals

The project is designed to solve a practical workflow:

1. User asks a question in plain English.
2. Model generates SQL from that question.
3. SQL is executed directly on a local database.
4. Results are shown in a readable table format.

This shortens the path from intent to query execution and can serve as a base for a larger NL-to-SQL product.

## 2. Repository Structure

```text
SQL-query-automation-platfrom/
|-- dataset/
|   `-- text_to_sql_dataset_5000.csv
|-- model/
|   |-- config.json
|   |-- generation_config.json
|   |-- model.safetensors
|   |-- tokenizer.json
|   `-- tokenizer_config.json
|-- results/
|   |-- checkpoint-500/
|   `-- checkpoint-508/
|-- build_database.py
|-- generate_sql.py
|-- train.py
|-- requirements.txt
`-- README.md
```

What each script does:

- `train.py`: Fine-tunes `t5-small` on your dataset and saves model/tokenizer.
- `generate_sql.py`: Loads the trained model, accepts interactive questions, generates SQL, executes SQL, and displays results.
- `build_database.py`: Creates a local SQLite database and inserts sample data.

## 3. Data and Learning Task

### Dataset contract

Training expects `dataset/text_to_sql_dataset_5000.csv` with at least:

- `question`: Natural language input.
- `sql`: Ground-truth SQL query.

### Task formulation

The model is trained as a sequence-to-sequence translation task using prompt-prefix conditioning:

```text
translate English to SQL: <question>
```

The same prefix must be used in both training and inference to keep behavior consistent.

## 4. Training Pipeline Deep Dive (`train.py`)

The training script performs these steps:

1. Loads CSV with Pandas.
2. Splits data with `train_test_split`:
	 - 10% test split first.
	 - 10% of the remaining training portion for validation.
3. Converts DataFrames to Hugging Face `Dataset` objects.
4. Tokenizes inputs and labels with max length 64, truncation, and max-length padding.
5. Trains `T5ForConditionalGeneration` using Hugging Face `Trainer`.
6. Saves model/tokenizer to `model/`.

### Key training settings currently used

- Base model: `t5-small`
- Epochs: `2`
- Batch size: `16` (per device)
- Save steps: `500`
- Logging steps: `50`
- Mixed precision: enabled automatically when CUDA is available (`fp16=True`)

### Important behavior notes

- The script creates `val_dataset` but does not pass it to `Trainer` as `eval_dataset`, so no validation metrics are reported during training by default.
- Labels are copied from tokenizer output IDs directly; this is standard for seq2seq fine-tuning with `Trainer`.

## 5. Inference and Execution Flow (`generate_sql.py`)

At runtime, the CLI does the following:

1. Detects device (`cuda` if available, else `cpu`).
2. Connects to SQLite database file `my_ai_database.db`.
3. Loads tokenizer and model from local model artifacts.
4. Accepts user commands/questions in a loop.
5. Generates SQL with beam search and decoding controls.
6. Executes generated SQL against SQLite.
7. Prints tabulated query results.

### Generation configuration in code

- `max_length=200`
- `num_beams=4`
- `early_stopping=True`
- `repetition_penalty=1.2`
- `length_penalty=1.0`

### Supported CLI commands

- `help`: Shows available commands.
- `history`: Displays prior natural-language questions in session memory.
- `clear`: Clears in-memory question history.
- `quit`: Closes DB connection and exits.

### Error handling

Inference catches two major error classes:

- Model generation/runtime exceptions.
- SQLite execution errors (invalid SQL, missing table/column, etc.).

This is useful because text-to-SQL systems can produce syntactically valid but schema-incompatible SQL.

## 6. Database Bootstrap (`build_database.py`)

The project includes a deterministic local DB setup:

- SQLite file: `my_ai_database.db`
- Tables created if missing:
	- `orders`
	- `employees`
	- `customers`
	- `students`
	- `products`
- Sample rows inserted with `INSERT OR IGNORE` to avoid duplicate-key failure on re-runs.

This script makes inference demos reproducible and provides immediate queryable data without external dependencies.

## 7. Installation

### Prerequisites

- Python 3.10+ recommended
- `pip`

### Setup

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 8. End-to-End Usage

### Step 1: (Optional) Train model

If you already have a valid `model/` folder, you can skip training.

```bash
python train.py
```

### Step 2: Build local SQLite database

```bash
python build_database.py
```

### Step 3: Start interactive generator

```bash
python generate_sql.py
```

## 9. Example Session

```text
Enter question: show all customers in new york
[Generated SQL]: SELECT * FROM customers WHERE city = 'New York';
[Generation time]: 0.42s

--- Database Results ---
+-------------+--------------+----------+-----+------------+
| customer_id | name         | city     | age | membership |
+-------------+--------------+----------+-----+------------+
| 101         | Diana Prince | New York | 28  | Gold       |
+-------------+--------------+----------+-----+------------+
------------------------
```

## 10. Dependencies

Installed from `requirements.txt`:

- `torch`
- `transformers`
- `datasets`
- `pandas`
- `scikit-learn`
- `tabulate`

## 11. Limitations and Practical Notes

- Accuracy is bounded by dataset quality and schema coverage.
- Generated SQL may be semantically wrong even if syntactically valid.
- Current workflow uses SQLite only; SQL dialect differences may appear when moving to other databases.
- Training currently does not report validation metrics unless `eval_dataset` and evaluation strategy are configured.

## 12. Suggested Next Improvements

1. Add evaluation during training (exact match, execution accuracy).
2. Add schema-aware prompting (inject table/column metadata in prompt).
3. Add SQL safety guardrails (allow-list operations, query timeout, read-only mode).
4. Add tests for preprocessing, generation, and DB execution behavior.
5. Add configurable DB backend support (SQLite/PostgreSQL/MySQL adapters).

## 13. Troubleshooting

- `no such table`: run `python build_database.py` to rebuild schema.
- Model load errors: ensure `model/` contains tokenizer and model artifacts.
- CUDA issues: verify PyTorch CUDA installation; CPU fallback is automatic.
- Empty/poor predictions: ensure training and inference use identical prompt prefix.
