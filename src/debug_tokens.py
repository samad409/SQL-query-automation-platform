import pickle

# Load your tokenizer
with open('../processed_data/output_tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

print("--- VOCABULARY CHECK ---")
print(f"Total words: {len(tokenizer.word_index)}")
print(f"Index 1: {tokenizer.index_word.get(1)}")
print(f"Index 2: {tokenizer.index_word.get(2)}")

# Check for start/end tokens
start_check = tokenizer.word_index.get('<start>') or tokenizer.word_index.get('start')
end_check = tokenizer.word_index.get('<end>') or tokenizer.word_index.get('end')

print(f"Start Token Index: {start_check}")
print(f"End Token Index: {end_check}")