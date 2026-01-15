# SQL Query Automation Platform

An end-to-end deep learning pipeline that converts natural language questions into SQL queries using a sequence-to-sequence (Seq2Seq) architecture.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Preprocessing Pipeline](#preprocessing-pipeline)
- [Usage](#usage)
- [Output Files](#output-files)
- [Technologies Used](#technologies-used)
- [Future Work](#future-work)

---

## 🎯 Overview

This platform automates the generation of SQL queries from natural language questions. It preprocesses text data through a multi-stage pipeline, preparing it for training a neural machine translation model. The goal is to enable users to query databases using plain English instead of writing SQL manually.

**Key Features:**
- Natural language to SQL translation
- Multi-stage preprocessing pipeline
- Tokenization with OOV (Out-Of-Vocabulary) handling
- Sequence padding and train/validation splitting
- Ready-to-use processed data for model training

---

## 📁 Project Structure

```
SQL-query-automation-platform/
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── data/
│   ├── train.csv                 # Training dataset (questions + SQL queries)
│   └── test.csv                  # Test dataset
├── processed_data/
│   ├── encoder_input_train.npy   # Padded training questions
│   ├── encoder_input_val.npy     # Padded validation questions
│   ├── decoder_input_train.npy   # Padded training SQL queries
│   ├── decoder_input_val.npy     # Padded validation SQL queries
│   ├── input_tokenizer.pickle    # Tokenizer for questions
│   └── output_tokenizer.pickle   # Tokenizer for SQL queries
└── src/
    ├── run_preprocessing.py      # Main preprocessing script
    └── preprocessing/
        ├── stage1_loader.py      # Data loading and cleaning
        ├── stage2_tokenizer.py   # Text tokenization
        └── stage3_padding.py     # Sequence padding and splitting
```

---

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SQL-query-automation-platform.git
   cd SQL-query-automation-platform
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset

The dataset consists of CSV files containing pairs of natural language questions and their corresponding SQL queries.

### Data Format

| Column | Description |
|--------|-------------|
| `question` | Natural language question (e.g., "Tell me what the notes are for South Australia") |
| `sql` | Corresponding SQL query (e.g., "SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA") |

### Example Samples

| Question | SQL Query |
|----------|-----------|
| Tell me what the notes are for South Australia | `SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA` |
| What is the format for South Australia? | `SELECT Format FROM table WHERE State/territory = South Australia` |

---

## ⚙️ Preprocessing Pipeline

The preprocessing is divided into three modular stages:

### Stage 1: Data Loading (`stage1_loader.py`)

- Loads training and test CSV files using Pandas
- Extracts `question` and `sql` columns
- Returns clean lists of questions and SQL queries

```python
questions, sql_queries = load_and_clean_data('train.csv', 'test.csv')
```

### Stage 2: Tokenization (`stage2_tokenizer.py`)

- Creates separate tokenizers for questions and SQL queries
- Converts text to integer sequences
- Handles Out-Of-Vocabulary (OOV) tokens with `<OOV>` placeholder
- Uses different filter settings for natural language vs SQL syntax

```python
input_seqs, output_seqs, input_tokenizer, output_tokenizer = tokenize_data(questions, sql_queries)
```

### Stage 3: Padding & Splitting (`stage3_padding.py`)

- Pads all sequences to a fixed length of 30 tokens
- Uses post-padding (zeros added at the end)
- Splits data into 80% training and 20% validation sets
- Uses a fixed random state for reproducibility

```python
x_train, x_val, y_train, y_val = pad_and_split(input_seqs, output_seqs)
```

---

## 🚀 Usage

### Run the Complete Preprocessing Pipeline

```bash
cd src
python run_preprocessing.py
```

### Expected Output

```
--- Stage 1: Loading Data from data/train.csv ---
Loaded X samples.
--- Stage 2: Tokenization ---
Tokenization complete.
Input Vocab Size: XXX
Output Vocab Size: XXX
--- Stage 3: Padding & Splitting ---
Padding Max Length: 30
Training Set Shape: (X, 30)
Validation Set Shape: (X, 30)

✅ All Preprocessing Stages Complete. Data saved to 'processed_data/'.
```

---

## 📦 Output Files

After running the preprocessing pipeline, the following files are generated in `processed_data/`:

| File | Description | Format |
|------|-------------|--------|
| `encoder_input_train.npy` | Tokenized & padded training questions | NumPy array |
| `encoder_input_val.npy` | Tokenized & padded validation questions | NumPy array |
| `decoder_input_train.npy` | Tokenized & padded training SQL queries | NumPy array |
| `decoder_input_val.npy` | Tokenized & padded validation SQL queries | NumPy array |
| `input_tokenizer.pickle` | Fitted tokenizer for questions | Pickle file |
| `output_tokenizer.pickle` | Fitted tokenizer for SQL queries | Pickle file |

### Loading Processed Data

```python
import numpy as np
import pickle

# Load numpy arrays
x_train = np.load('processed_data/encoder_input_train.npy')
y_train = np.load('processed_data/decoder_input_train.npy')

# Load tokenizers
with open('processed_data/input_tokenizer.pickle', 'rb') as handle:
    input_tokenizer = pickle.load(handle)
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **Pandas** | Data loading and manipulation |
| **NumPy** | Numerical operations and array handling |
| **TensorFlow/Keras** | Tokenization and sequence preprocessing |
| **Scikit-learn** | Train/validation data splitting |

---

## 🔮 Future Work

- [ ] Implement Seq2Seq model with attention mechanism
- [ ] Add LSTM/GRU encoder-decoder architecture
- [ ] Create inference pipeline for real-time predictions
- [ ] Build chatbot interface for SQL generation
- [ ] Add support for multiple SQL dialects
- [ ] Implement beam search for better query generation
- [ ] Add evaluation metrics (BLEU score, exact match accuracy)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

*Built with ❤️ for automating SQL query generation*