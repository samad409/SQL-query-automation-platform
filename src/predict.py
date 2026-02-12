import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
MODEL_PATH = 'sql_automation_model.h5'
TOKENIZER_DIR = '../processed_data'
MAX_LEN_EN = 30
MAX_LEN_SQL = 30

def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(f'{TOKENIZER_DIR}/input_tokenizer.pickle', 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open(f'{TOKENIZER_DIR}/output_tokenizer.pickle', 'rb') as f:
        output_tokenizer = pickle.load(f)
    return model, input_tokenizer, output_tokenizer

def clean_text(text):
    # REMOVED the aggressive regex that strips < > symbols
    text = text.lower().strip()
    return text

def decode_sequence(input_text, model, input_tokenizer, output_tokenizer):
    # 1. Preprocess Question
    text = clean_text(input_text)
    seq = input_tokenizer.texts_to_sequences([text])
    encoder_input = pad_sequences(seq, maxlen=MAX_LEN_EN, padding='post')

    # 2. Find Start/End Tokens
    # We try both '<start>' and 'start' to be safe
    start_token = output_tokenizer.word_index.get('<start>')
    if start_token is None:
        start_token = output_tokenizer.word_index.get('start')
    
    end_token = output_tokenizer.word_index.get('<end>')
    if end_token is None:
        end_token = output_tokenizer.word_index.get('end')

    if start_token is None:
        return "Error: Could not find start token in vocabulary."

    # 3. Initialize Decoder
    decoder_input = np.zeros((1, MAX_LEN_SQL))
    decoder_input[0, 0] = start_token

    decoded_sentence = []

    # 4. Generate Loop
    for i in range(1, MAX_LEN_SQL):
        # Predict
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        
        # Get the token with highest probability for the current step
        # (We look at index i-1 because that matches the current step in the sequence)
        sampled_token_index = np.argmax(predictions[0, i-1, :])
        
        # If the model predicts padding (0), it usually means it's done or confused
        if sampled_token_index == 0:
            break

        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '?')

        # Stop conditions
        if sampled_token_index == end_token or sampled_word == 'end' or sampled_word == '<end>':
            break
            
        decoded_sentence.append(sampled_word)
        
        # Update input for next step
        decoder_input[0, i] = sampled_token_index

    return ' '.join(decoded_sentence)

if __name__ == "__main__":
    model, input_tokenizer, output_tokenizer = load_resources()
    print("\n--- CHATBOT READY ---")
    
    while True:
        q = input("Question: ")
        if q == 'exit': break
        
        # 1. Generate the raw prediction
        sql_query = decode_sequence(q, model, input_tokenizer, output_tokenizer)
        
        # 2. Apply the "Missing Equals Sign" Fix
        if "where" in sql_query and "=" not in sql_query:
            # Quick hack: find the last word (the value) and put an = before it
            parts = sql_query.split()
            if len(parts) > 1:
                # Reassemble: ... where col = value
                sql_query = " ".join(parts[:-1]) + " = " + parts[-1]

        # 3. Print final result
        print("SQL:", sql_query)
        print("-" * 20)