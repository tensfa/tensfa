from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM
import numpy as np

maxlen = 100
vocab_size = 43

embeddings = np.array(np.random.random((412457, 400)), dtype=np.float32)

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embeddings], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(128))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])