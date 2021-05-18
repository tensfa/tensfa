import numpy as np
import pandas as pd
from preprocess import DataLoader

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, Dropout, MaxPooling3D
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import optimizers

target_width = 160
target_height = 192
target_depth = 192

num_classes = 3
batch_size = 4

data_loader = DataLoader(target_shape=(target_width, target_height, target_depth))
train, test = data_loader.Get_Data_List()

print("Train size: " + str(len(train)))
print("Test size: " + str(len(test)))

def custom_one_hot(labels):
  label_dict = {"stableAD":np.array([0,0,1]),
              "stableMCI":np.array([0,1,0]),
              "stableNL":np.array([1,0,0])}
  encoded_labels = []
  for label in labels:
    encoded_labels.append(label_dict[label].reshape(1,3))
  return np.asarray(encoded_labels)

def additional_data_prep(train, test):
  # Extract data from tuples
  train_labels, train_data = zip(*train)
  test_labels, test_data = zip(*test)
  X_train = np.asarray(train_data)
  X_test = np.asarray(test_data)
  y_train = custom_one_hot(train_labels)
  y_test = custom_one_hot(test_labels)
  return X_train, y_train, X_test, y_test

X, y, X_test, y_test = additional_data_prep(train, test)

X = np.expand_dims(X, axis=-1).reshape((X.shape[0],target_width,target_height,target_depth,1))
X_test = np.expand_dims(X_test, axis=-1).reshape((X_test.shape[0],target_width,target_height,target_depth,1))

model = Sequential()
model.add(Conv3D(24, kernel_size=(13, 11, 11), activation='relu', input_shape=(target_width,target_height,target_depth,1), padding='same', strides=4))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
model.add(Dropout(0.1))
model.add(Conv3D(48, kernel_size=(6, 5, 5), activation='relu', padding='same'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
model.add(Dropout(0.1))
model.add(Conv3D(24, kernel_size=(4, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(3, 3, 3), strides=2))
model.add(Dropout(0.1))
model.add(Conv3D(8, kernel_size=(2, 2, 2), activation='relu'))
model.add(MaxPooling3D(pool_size=(1, 1, 1), strides=2))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(learning_rate=0.0015),
              metrics=['accuracy','categorical_crossentropy'])

model.fit(X, y, batch_size=batch_size, epochs=10, verbose=2, use_multiprocessing=True)

model.evaluate(X_test, y_test, verbose=2, use_multiprocessing=True)