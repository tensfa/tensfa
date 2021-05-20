
import numpy as np
import tensorflow.keras.layers as layers
from tensorflow import keras

NUM_LABELS = 11
train_data = np.random.random(size=(100, 20, 130))

train_data = np.expand_dims(train_data, axis=3)

# generate one-hot random vector
train_labels =  np.eye(11)[np.random.choice(1, 100)]

model = keras.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=train_data.shape[1:]))

model.add(layers.MaxPool2D(2,2))
model.add(layers.Conv2D(32, (3,3), activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(34, activation="relu"))
model.add(layers.Dense(NUM_LABELS))
model.summary()

model.compile(
   loss = 'categorical_crossentropy', optimizer = 'sgd', metrics = ['accuracy']
)

history = model.fit(train_data , train_labels, epochs=1, validation_split=0.1)