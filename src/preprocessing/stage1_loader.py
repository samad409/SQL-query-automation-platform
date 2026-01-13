import pandas as pd

def load_and_clean_data(train_path, test_path):
    """
    Stage 1: Load datasets and separate questions from SQL queries.
    """
    print(f"--- Stage 1: Loading Data from {train_path} ---")
    
    # Load the datasets 
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Combine them for building a complete vocabulary (optional but recommended)
    # or just return the training data as per your specific flow.
    # For this structure, we'll process the training file primarily.
    
    # Extraction [cite: 49]
    # Ensure your CSV headers match these keys ('question', 'sql')
    questions = df_train['question'].astype(str).tolist()
    sql_queries = df_train['sql'].astype(str).tolist()
    
    print(f"Loaded {len(questions)} samples.")
    return questions, sql_queries