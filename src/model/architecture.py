import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_attention_model(vocab_size_en, vocab_size_sql, max_len_en, max_len_sql, embedding_dim=128, lstm_units=256):
    """
    Builds the Seq2Seq model with Attention as described in the paper.
    """
    
    # --- 1. ENCODER (The "Reader") ---
    # Input: Natural Language Question (from encoder_input_train.npy)
    encoder_inputs = Input(shape=(max_len_en,), name='encoder_inputs')
    
    # Embedding: Converts words to vectors [cite: 91]
    enc_emb = layers.Embedding(input_dim=vocab_size_en, output_dim=embedding_dim, name='enc_embedding')(encoder_inputs)
    
    # LSTM: Processes the sequence. 
    # return_sequences=True is needed for Attention.
    # return_state=True is needed to initialize the Decoder.
    encoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True, name='encoder_lstm')
    encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
    encoder_states = [state_h, state_c] # Context vector [cite: 125]

    # --- 2. DECODER (The "Writer") ---
    # Input: SQL Query (from decoder_input_train.npy)
    decoder_inputs = Input(shape=(max_len_sql,), name='decoder_inputs')
    
    # Embedding for SQL
    dec_emb_layer = layers.Embedding(input_dim=vocab_size_sql, output_dim=embedding_dim, name='dec_embedding')
    dec_emb = dec_emb_layer(decoder_inputs)
    
    # LSTM: Generates the output. We initialize it with the Encoder's states.
    decoder_lstm = layers.LSTM(lstm_units, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

    # --- 3. ATTENTION MECHANISM [cite: 73] ---
    # The Attention layer looks at the Decoder's current state and compares it 
    # to all Encoder outputs to decide what to focus on.
    attention_layer = layers.Attention(name='attention_layer')
    attention_result = attention_layer([decoder_outputs, encoder_outputs])

    # Combine Attention output with Decoder output
    decoder_concat_input = layers.Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attention_result])

    # --- 4. OUTPUT LAYER ---
    # Softmax to predict the next word in the vocabulary [cite: 78]
    dense = layers.Dense(vocab_size_sql, activation='softmax', name='output_dense')
    decoder_pred = dense(decoder_concat_input)

    # Compile the model
    model = models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
    
    # Compile with Adam (lr=0.001) and Sparse Categorical Crossentropy [cite: 89, 92]
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model