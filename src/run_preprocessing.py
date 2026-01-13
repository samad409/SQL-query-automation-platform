import os
import numpy as np
import pickle

# Import our stages
from preprocessing.stage1_loader import load_and_clean_data
from preprocessing.stage2_tokenizer import tokenize_data
from preprocessing.stage3_padding import pad_and_split

def main():
    # Define Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
    
    # 1. Run Stage 1
    questions, sql_queries = load_and_clean_data(
        os.path.join(DATA_DIR, 'train.csv'),
        os.path.join(DATA_DIR, 'test.csv')
    )
    
    # 2. Run Stage 2
    input_seqs, output_seqs, in_token, out_token = tokenize_data(questions, sql_queries)
    
    # 3. Run Stage 3
    x_train, x_val, y_train, y_val = pad_and_split(input_seqs, output_seqs)
    
    # Save the processed data for the Model to use later
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    np.save(os.path.join(PROCESSED_DIR, 'encoder_input_train.npy'), x_train)
    np.save(os.path.join(PROCESSED_DIR, 'encoder_input_val.npy'), x_val)
    np.save(os.path.join(PROCESSED_DIR, 'decoder_input_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'decoder_input_val.npy'), y_val)
    
    # Save Tokenizers (Critical for the Chatbot later!)
    with open(os.path.join(PROCESSED_DIR, 'input_tokenizer.pickle'), 'wb') as handle:
        pickle.dump(in_token, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(PROCESSED_DIR, 'output_tokenizer.pickle'), 'wb') as handle:
        pickle.dump(out_token, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n✅ All Preprocessing Stages Complete. Data saved to 'processed_data/'.")

if __name__ == "__main__":
    main()