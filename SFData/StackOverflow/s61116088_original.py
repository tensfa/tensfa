

Here is my code:

# data consists of 1 dimensional data (3277 elements). Number of data is 439
train_data = .... # numpy.ndarray
# I would like to classify data into 5 classes.
train_labels = .... # numpy.ndarray

print(train_data.shape) # -> Shape of train_data: (439, 3277)
print('Shape of train_labels:', train_labels.shape) # -> Shape of train_labels: (439,)
# prepare 5 one hot encoding array
categorical_labels = to_categorical(train_labels, 5)
print('Shape of categorical_labels:', categorical_labels.shape) # -> Shape of categorical_labels: (439, 5)

# I make a model to have 3277-elements data and classify data into 5 labels.
model = keras.Sequential([
    keras.layers.Dense(30, activation='relu', input_shape=(3277,)),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(5, activation='softmax')
])
model.summary()
model.compile(optimizer='adam',
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy'])
model.fit(data, categorical_labels, epochs=5, verbose=1) # A
#model.fit(data, train_labels, epochs=5, verbose=1) # B