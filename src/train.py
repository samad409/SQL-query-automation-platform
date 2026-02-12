import numpy as np
import pickle
import os
from model.architecture import build_attention_model

# --- CONFIGURATION ---
DATA_DIR = '../processed_data' # Pointing to the folder in your screenshot
BATCH_SIZE = 16  # As specified in paper [cite: 79]
EPOCHS = 5       # As specified in paper [cite: 79]

def load_data():
    print("Loading data from:", DATA_DIR)
    
    # Load the .npy files seen in your screenshot
    encoder_input_train = np.load(os.path.join(DATA_DIR, 'encoder_input_train.npy'))
    decoder_input_train = np.load(os.path.join(DATA_DIR, 'decoder_input_train.npy'))
    encoder_input_val = np.load(os.path.join(DATA_DIR, 'encoder_input_val.npy'))
    decoder_input_val = np.load(os.path.join(DATA_DIR, 'decoder_input_val.npy'))

    # Load tokenizers to get vocabulary sizes
    with open(os.path.join(DATA_DIR, 'input_tokenizer.pickle'), 'rb') as f:
        input_tokenizer = pickle.load(f)
    with open(os.path.join(DATA_DIR, 'output_tokenizer.pickle'), 'rb') as f:
        output_tokenizer = pickle.load(f)
        
    vocab_size_en = len(input_tokenizer.word_index) + 1
    vocab_size_sql = len(output_tokenizer.word_index) + 1
    
    return (encoder_input_train, decoder_input_train, encoder_input_val, decoder_input_val, 
            vocab_size_en, vocab_size_sql)

def create_targets(decoder_input):
    """
    Creates targets by shifting decoder input by 1.
    Input:  [<start>, SELECT, *]
    Target: [SELECT, *, <end>]
    """
    # This assumes your padding is 0. 
    # We create a new array where everything is shifted left by 1.
    targets = np.zeros_like(decoder_input)
    targets[:, :-1] = decoder_input[:, 1:] 
    return targets

# --- MAIN EXECUTION ---
# 1. Load Data
enc_train, dec_train, enc_val, dec_val, vocab_en, vocab_sql = load_data()

# 2. Prepare Targets (Shifted SQL)
# The model tries to predict the NEXT word, so target is decoder_input shifted by 1
target_train = create_targets(dec_train)
target_val = create_targets(dec_val)

# 3. Build Model
max_len_en = enc_train.shape[1]
max_len_sql = dec_train.shape[1]

print(f"Building Model... (Vocab EN: {vocab_en}, Vocab SQL: {vocab_sql})")
model = build_attention_model(vocab_en, vocab_sql, max_len_en, max_len_sql)
model.summary()

# 4. Train
print("Starting Training...")
history = model.fit(
    [enc_train, dec_train],  # Inputs: (Question, Current SQL)
    target_train,            # Output: (Next SQL Token)
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=([enc_val, dec_val], target_val)
)

# 5. Save the trained model
model.save('sql_automation_model.h5')
print("Model saved as sql_automation_model.h5")