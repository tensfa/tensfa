import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Sequential

# create data as in your matlab data
n_class = 6
n_sample = 6500
X_train = np.random.uniform(0,1, (3,12,n_sample)) # (3,12,n_sample)
Y_train = tf.keras.utils.to_categorical(np.random.randint(0,n_class, n_sample)) # (n_sample, n_classes)

# reshape your data for conv2d
X_train = X_train.transpose(2,0,1) # (n_sample,3,12)

classifier = Sequential()
classifier.add(Conv2D(8, kernel_size=(3,3) , input_shape=(3, 12, 1), padding="same"))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))

classifier.add(Conv2D(8, kernel_size=(3,3), padding="same"))
classifier.add(BatchNormalization())
classifier.add(Activation('relu'))

classifier.add(Flatten())
classifier.add(Dense(8, activation='relu'))
classifier.add(Dense(n_class, activation='softmax'))
classifier.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])

history = classifier.fit(X_train, Y_train, batch_size = 32, epochs=2, verbose=2)
