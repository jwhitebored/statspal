INPUT_SEQUENCE_LENGTH = 1024
# LSTM input shape convention: (sequence_length, features_per_step)
INPUT_SHAPE = (INPUT_SEQUENCE_LENGTH, 1)

model = keras.Sequential()
# Input Layer: Must accept (1024, 1) sized sequences
model.add(keras.Input(shape=INPUT_SHAPE))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, return_sequences=True, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(256, activation='tanh', kernel_initializer='he_normal'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(21))
print(model.summary())

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), #Turns out this version of Keras doesn't have AdamW: AttributeError: module 'keras.api._v2.keras.optimizers' has no attribute 'AdamW'
    #optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004,), #Will switch to AdamW once build is updated
    metrics=["accuracy"]
)
