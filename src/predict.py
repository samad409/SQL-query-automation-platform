import numpy as np
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- CONFIGURATION ---
MODEL_PATH = 'sql_automation_model.h5'
TOKENIZER_DIR = '../processed_data'
MAX_LEN_EN = 30  # Must match your training value [cite: 62]
MAX_LEN_SQL = 30 # Must match your training value [cite: 62]

def load_resources():
    print("Loading resources...")
    # 1. Load the trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # 2. Load the tokenizers (to convert text <-> numbers)
    with open(f'{TOKENIZER_DIR}/input_tokenizer.pickle', 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open(f'{TOKENIZER_DIR}/output_tokenizer.pickle', 'rb') as f:
        output_tokenizer = pickle.load(f)
        
    return model, input_tokenizer, output_tokenizer

def clean_text(text):
    """
    Applies the same cleaning as your training data 
    """
    text = text.lower()
    text = re.sub(r"([?.!,])", r" \1 ", text)
    text = re.sub(r'[" "]+', " ", text)
    text = re.sub(r"[^a-zA-Z?.!,0-9]+", " ", text)
    text = text.strip()
    return text

def decode_sequence(input_text, model, input_tokenizer, output_tokenizer):
    """
    The core logic: Generates SQL one word at a time 
    """
    # 1. Preprocess the User Input (English)
    text = clean_text(input_text)
    seq = input_tokenizer.texts_to_sequences([text])
    encoder_input = pad_sequences(seq, maxlen=MAX_LEN_EN, padding='post')

    # 2. Initialize the Decoder Input
    # Start with the specific start token (usually '<start>')
    start_token = output_tokenizer.word_index.get('<start>')
    if start_token is None:
        # Fallback if specific token not found (common in some setups)
        start_token = output_tokenizer.word_index.get('start', 1) 
        
    decoder_input = np.zeros((1, MAX_LEN_SQL))
    decoder_input[0, 0] = start_token

    # 3. Generate Loop
    decoded_sentence = []
    
    for i in range(1, MAX_LEN_SQL):
        # Predict the next token
        # We feed: [Question, Current_Partial_SQL]
        predictions = model.predict([encoder_input, decoder_input], verbose=0)
        
        # The prediction for the current step is the token with highest probability
        # We look at the i-1 index because we are predicting the *next* token
        sampled_token_index = np.argmax(predictions[0, i-1, :])
        
        # Convert index back to word
        sampled_word = output_tokenizer.index_word.get(sampled_token_index, '?')
        
        # Stop if we hit the end token
        if sampled_word == '<end>' or sampled_word == 'end':
            break
            
        decoded_sentence.append(sampled_word)
        
        # Update decoder input for the next loop iteration
        decoder_input[0, i] = sampled_token_index

    return ' '.join(decoded_sentence)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load everything
    model, input_tokenizer, output_tokenizer = load_resources()
    
    print("\n--- SQL AUTOMATION CHATBOT READY ---")
    print("Type 'exit' to quit.\n")
    
    while True:
        # Get User Input 
        user_input = input("Enter your question: ")
        
        if user_input.lower() == 'exit':
            break
            
        try:
            # Generate SQL
            sql_query = decode_sequence(user_input, model, input_tokenizer, output_tokenizer)
            
            # Display Output 
            print(f"Generated SQL: {sql_query}")
            print("-" * 30)
            
        except Exception as e:
            print(f"Error: {e}")