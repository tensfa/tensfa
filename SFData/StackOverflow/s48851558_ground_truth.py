from tensorflow import keras
import tensorflow as tf
import numpy as np

from tensorflow.keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results.astype('float32')

# Our vectorized training data
x_train = vectorize_sequences(train_data)

# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
y_train = y_train.reshape((-1, 1))
y_test = y_test.reshape((-1, 1))

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

model = keras.models.Sequential()
model.add(keras.layers.Dense(16, activation='relu', input_shape=(10000,), name='reviews'))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
estimator_model = keras.estimator.model_to_estimator(keras_model=model)

def input_function(features,labels=None,shuffle=False,epochs=None,batch_size=None):
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"reviews_input": features},
        y=labels,
        shuffle=shuffle,
        num_epochs=epochs,
        batch_size=batch_size
    )
    return input_fn

estimator_model.train(input_fn=input_function(partial_x_train, partial_y_train, True, 1, 512))
score = estimator_model.evaluate(input_function(x_val, y_val, False, 1, 512))
print(score)
