import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

dataset = np.random.randint(0, 10, (51, 6), dtype=np.int64)

results = tf.data.Dataset.from_tensor_slices(dataset)
sequences = results.batch(51, drop_remainder=True)
def split(batch):
    input_ = batch[:-1]
    output_ = batch[1:]
    return input_, output_
dataset = sequences.map(split)

model = keras.Sequential()
model.add(layers.Embedding(input_dim=64, output_dim=32))
model.add(layers.GRU(128, return_sequences=True))
model.add(layers.SimpleRNN(32))
model.add(layers.Dense(100, activation='softmax'))

model.compile(
    optimizer='adam',
    loss=keras.losses.categorical_crossentropy,
    metrics=['accuracy']
)

model.fit(dataset, epochs=1)