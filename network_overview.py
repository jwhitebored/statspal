# --- 1. Define Constants ---
NUM_CLASSES = 21 # 0 through 20
INPUT_SEQUENCE_LENGTH = 1024
INPUT_SHAPE = (INPUT_SEQUENCE_LENGTH, 1)
LSTM_UNITS = 256

# --- 2. Define the input tensor ---
inputs = Input(shape=INPUT_SHAPE, name='input_sequence')

# --- 3. Initial Feature Extractor (LSTM 1) ---
# Converts (1024, 1) -> (1024, 256)
x = LSTM(LSTM_UNITS, return_sequences=True, activation='tanh', kernel_initializer='he_normal', name='lstm_1')(inputs)
x = Dropout(0.2, name='dropout_1')(x)
l_prev = x # l_prev is now the tensor for the first residual addition


# --- 4. Residual Block 1 (LSTM 2) ---
x = LSTM(LSTM_UNITS, return_sequences=True, activation='tanh', kernel_initializer='he_normal', name='lstm_2_main')(l_prev)
x = Dropout(0.2, name='dropout_2')(x)
l_prev = Add(name='skip_add_2')([l_prev, x])


# --- 5. Residual Block 2 (LSTM 3) ---
x = LSTM(LSTM_UNITS, return_sequences=True, activation='tanh', kernel_initializer='he_normal', name='lstm_3_main')(l_prev)
x = Dropout(0.2, name='dropout_3')(x)
l_prev = Add(name='skip_add_3')([l_prev, x]) # l_prev is now the input to Block 4


# --- IMPLEMENTING LONG SKIP CONNECTION HERE ---
# The tensor 'l_prev' is currently the input to the LSTM 4 block.
# We must add 'inputs' (1024, 1) to 'l_prev' (1024, 256).

# 1. Projection Shortcut: Use Conv1D(1x1) to convert input depth from 1 to 256
skip_projection_H4 = Conv1D(
    filters=LSTM_UNITS, 
    kernel_size=1, 
    padding='same', 
    name='input_to_H4_projection'
)(inputs)

# 2. Merge: Add the current block input (l_prev) and the projected original input
l_H4_new_input = Add(name='long_skip_merge_H4')([l_prev, skip_projection_H4])
# l_H4_new_input is the new tensor feeding into LSTM 4


# --- 6. Residual Block 3 (LSTM 4) ---
# Now use the newly merged tensor as input
x = LSTM(LSTM_UNITS, return_sequences=True, activation='tanh', kernel_initializer='he_normal', name='lstm_4_main')(l_H4_new_input)
x = Dropout(0.2, name='dropout_4')(x)
# The residual addition needs to use the *input* to the LSTM block, which is l_H4_new_input
l_prev = Add(name='skip_add_4')([l_H4_new_input, x])


# --- 7. Residual Block 4 (LSTM 5) ---
x = LSTM(LSTM_UNITS, return_sequences=True, activation='tanh', kernel_initializer='he_normal', name='lstm_5_main')(l_prev)
x = Dropout(0.2, name='dropout_5')(x)
l_final_sequence = Add(name='skip_add_5')([l_prev, x])


# --- 8. Final Sequence Reduction (LSTM 6) ---
# The last LSTM has return_sequences=False
x = LSTM(LSTM_UNITS, activation='tanh', kernel_initializer='he_normal', name='lstm_6_final')(l_final_sequence)
x = Dropout(0.2, name='dropout_6')(x)


# --- 9. Output Layer with Softmax ---
outputs = Dense(NUM_CLASSES, activation='softmax', name='output_softmax')(x)


# --- 10. Create and compile the final model ---
model = models.Model(inputs=inputs, outputs=outputs, name='residual_lstm_network_long_skip')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), #Turns out this version of TensorFlow keras doesn't have AdamW: AttributeError: module 'keras.api._v2.keras.optimizers' has no attribute 'AdamW'
    #optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004,), #clipnorm=1.0), #Switched to AdamW at reccomendation of Yacine
    metrics=["accuracy"]
)
