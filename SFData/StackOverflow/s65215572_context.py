from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
import numpy as np
from tensorflow.keras.utils import to_categorical

max_words = 100
embedding_dim = 100
maxlen = 100
total_samples = 100
training_samples = 70
validation_samples = 30
n_class = 16

data = np.array(np.random.random((total_samples, maxlen)), dtype=np.float32)
labels = np.random.randint(0, n_class, total_samples, dtype=np.int32)

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.summary()

model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy'])
history = model.fit(x_train, y_train, epochs=1, batch_size=32, validation_data=(x_val, y_val))