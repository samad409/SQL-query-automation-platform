import numpy as np
import pickle
import os

# --- PATHS ---
DATA_DIR = '../processed_data'

def inspect_data():
    print("--- INSPECTING TRAINING DATA ---")
    
    # 1. Load the Tokenizer
    with open(os.path.join(DATA_DIR, 'output_tokenizer.pickle'), 'rb') as f:
        tokenizer = pickle.load(f)
        
    # 2. Load the actual Training Data
    decoder_input = np.load(os.path.join(DATA_DIR, 'decoder_input_train.npy'))
    
    # 3. Get the first sample
    first_seq = decoder_input[0]
    
    print(f"\nFirst Sequence (Indices): {first_seq[:10]}")
    
    # 4. Decode it back to words
    words = []
    for idx in first_seq:
        if idx == 0: continue # Skip padding
        word = tokenizer.index_word.get(idx, '?')
        words.append(f"{word}({idx})")
        
    print(f"First Sequence (Words): {' '.join(words[:10])}")
    
    # 5. Check specific tokens
    print("\n--- TOKEN CHECK ---")
    print(f"Index for 'start': {tokenizer.word_index.get('start')}")
    print(f"Index for '<start>': {tokenizer.word_index.get('<start>')}")
    print(f"Index for 'select': {tokenizer.word_index.get('select')}")

if __name__ == "__main__":
    inspect_data()