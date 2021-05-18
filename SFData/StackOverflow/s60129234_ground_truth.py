import numpy as np
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, Conv3D, Dropout, MaxPooling3D
from tensorflow.keras.models import Sequential

target_width = 160
target_height = 192
target_depth = 192

num_classes = 3
batch_size = 4

def custom_one_hot(labels):
  label_dict = {"stableAD":np.array([0,0,1]),
              "stableMCI":np.array([0,1,0]),
              "stableNL":np.array([1,0,0])}
  encoded_labels = []
  for label in labels:
    encoded_labels.append(label_dict[label].reshape(1,3))
  return np.asarray(encoded_labels)

X = np.array(np.random.random((8, target_width, target_height, target_depth)), dtype=np.float32)
y = np.random.choice(["stableAD", "stableMCI", "stableNL"], 8)
y = custom_one_hot(y)
y = np.squeeze(y, 1)

X_test = np.array(np.random.random((3, target_width, target_height, target_depth)), dtype=np.float32)
y_test = np.random.choice(["stableAD", "stableMCI", "stableNL"], 3)
y_test = custom_one_hot(y_test)
y_test = np.squeeze(y_test, 1)

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

model.fit(X, y, batch_size=batch_size, epochs=1, verbose=2, use_multiprocessing=True)

model.evaluate(X_test, y_test, verbose=2, use_multiprocessing=True)