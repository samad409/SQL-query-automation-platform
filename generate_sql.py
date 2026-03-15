import torch
import logging
import time
import sqlite3
from tabulate import tabulate
from transformers import T5Tokenizer, T5ForConditionalGeneration

VERSION = "1.0.0"
MODEL_PATH = "model"
TOKENIZER_NAME = "t5-small"
PREFIX = "translate English to SQL: "
DB_NAME = "my_ai_database.db"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Database Connection Setup ---
print(f"Connecting to database '{DB_NAME}'...")
try:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    print("Database connected successfully.")
except sqlite3.Error as e:
    print(f"Failed to connect to database: {e}")
    exit(1)
# ---------------------------------

print(f"SQL Query Automation Platform v{VERSION}")
print("Loading trained model...")
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model = model.to(device)
print(f"Model loaded on {device}. Type 'help' for available commands.")

history = []

while True:
    question = input("\nEnter question: ").strip()
    
    if not question:
        continue
        
    if question.lower() == "quit":
        print("Closing database connection...")
        conn.close() 
        print("Goodbye!")
        break
        
    if question.lower() == "help":
        print("Commands: quit, help, history, clear")
        continue
        
    if question.lower() == "history":
        for i, q in enumerate(history, 1):
            print(f"{i}. {q}")
        continue
        
    if question.lower() == "clear":
        history.clear()
        print("History cleared.")
        continue
        
    history.append(question)
    logging.info(f"Query: {question}")
    
    input_text = PREFIX + question
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    input_ids = input_ids.to(device)
    
    start_time = time.time()
    
    try:
        # Generate the SQL string
        outputs = model.generate(
            input_ids, max_length=200, num_beams=4,
            early_stopping=True, repetition_penalty=1.2, length_penalty=1.0
        )
        elapsed = time.time() - start_time
        sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n[Generated SQL]: {sql}")
        print(f"[Generation time]: {elapsed:.2f}s")
        
        # --- Database Execution & Table Formatting ---
        print("\n--- Database Results ---")
        try:
            cursor.execute(sql)
            results = cursor.fetchall()
            
            if not results:
                print("Query executed successfully, but no data was returned.")
            else:
                # Extract headers directly from the database schema
                headers = [description[0] for description in cursor.description]
                
                # Print the data as a clean grid
                print(tabulate(results, headers=headers, tablefmt="grid"))
                
        except sqlite3.Error as db_error:
            # Catches syntax errors or hallucinated table/column names
            print(f"Database Execution Error: {db_error}")
        print("------------------------\n")
        # ---------------------------------------------

    except Exception as e:
        logging.error(f"Generation failed: {e}")