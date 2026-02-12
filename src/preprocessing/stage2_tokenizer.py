from tensorflow.keras.preprocessing.text import Tokenizer

def tokenize_data(questions, sql_queries):
    """
    Stage 2: Tokenize questions and SQL queries separately.
    Handles Out-Of-Vocabulary (OOV) tokens.
    """
    print("--- Stage 2: Tokenization ---")
    
    # --- 1. Process INPUT (Questions) ---
    # We remove < and > from filters just in case, though usually not needed for English
    input_tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
    input_tokenizer.fit_on_texts(questions)
    input_sequences = input_tokenizer.texts_to_sequences(questions)
    
    # --- 2. Process OUTPUT (SQL) ---
    # CRITICAL FIX: Add <start> and <end> tokens to every SQL query
    # If the query is "SELECT * FROM table", it becomes "<start> SELECT * FROM table <end>"
    processed_sql_queries = []
    for sql in sql_queries:
        processed_sql_queries.append(f"<start> {sql} <end>")
    
    # Tokenizer for SQL Queries
    # We explicitly EXCLUDE < and > from filters so the tokenizer doesn't strip them
    output_tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n') 
    
    output_tokenizer.fit_on_texts(processed_sql_queries)
    output_sequences = output_tokenizer.texts_to_sequences(processed_sql_queries)
    
    print("Tokenization complete.")
    print(f"Input Vocab Size: {len(input_tokenizer.word_index)}")
    print(f"Output Vocab Size: {len(output_tokenizer.word_index)}")
    
    # Debug: Print the first converted sequence to verify the fix
    print(f"DEBUG Sample SQL: {processed_sql_queries[0]}")
    print(f"DEBUG Sample Seq: {output_sequences[0]}")
    
    return input_sequences, output_sequences, input_tokenizer, output_tokenizer