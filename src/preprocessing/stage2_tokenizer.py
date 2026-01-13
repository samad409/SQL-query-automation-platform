from tensorflow.keras.preprocessing.text import Tokenizer

def tokenize_data(questions, sql_queries):
    """
    Stage 2: Tokenize questions and SQL queries separately.
    Handles Out-Of-Vocabulary (OOV) tokens.
    """
    print("--- Stage 2: Tokenization ---")
    
    # Tokenizer for Natural Language Questions [cite: 57]
    # 'oov_token' handles words not in the training vocabulary [cite: 59]
    input_tokenizer = Tokenizer(oov_token="<OOV>", filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    input_tokenizer.fit_on_texts(questions)
    input_sequences = input_tokenizer.texts_to_sequences(questions)
    
    # Tokenizer for SQL Queries [cite: 57]
    # We use a different tokenizer because SQL syntax is different from English
    output_tokenizer = Tokenizer(oov_token="<OOV>", filters='') # keep special chars for SQL
    output_tokenizer.fit_on_texts(sql_queries)
    output_sequences = output_tokenizer.texts_to_sequences(sql_queries)
    
    print("Tokenization complete.")
    print(f"Input Vocab Size: {len(input_tokenizer.word_index)}")
    print(f"Output Vocab Size: {len(output_tokenizer.word_index)}")
    
    return input_sequences, output_sequences, input_tokenizer, output_tokenizer