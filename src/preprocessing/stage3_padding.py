import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def pad_and_split(input_seqs, output_seqs):
    """
    Stage 3: Pad sequences to uniform length and split into Train/Val sets.
    """
    print("--- Stage 3: Padding & Splitting ---")
    
    MAX_LENGTH = 30  # Fixed sequence length as per paper [cite: 62]
    
    # Pad sequences [cite: 60]
    encoder_input_data = pad_sequences(input_seqs, maxlen=MAX_LENGTH, padding='post')
    decoder_input_data = pad_sequences(output_seqs, maxlen=MAX_LENGTH, padding='post')
    
    # Split 80% Training, 20% Validation [cite: 64]
    input_train, input_val, output_train, output_val = train_test_split(
        encoder_input_data, 
        decoder_input_data, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"Padding Max Length: {MAX_LENGTH}")
    print(f"Training Set Shape: {input_train.shape}")
    print(f"Validation Set Shape: {input_val.shape}")
    
    return input_train, input_val, output_train, output_val