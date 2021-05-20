import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPool2D
from tensorflow.keras.models import Sequential

mnist = input_data.read_data_sets(os.path.dirname(os.path.realpath(__file__))+"/../data/MNIST_data/", one_hot=False)
X, y = mnist.train.next_batch(55000)
y = np.eye(10)[y]

X=X/255.
X=np.reshape(X,(-1,28,28,1))

model=Sequential()
model.add(Conv2D(input_shape=(28,28,1),filters=16,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Conv2D(filters=16,kernel_size=(3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D())

model.add(Flatten())
model.add(Dense(units=400,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10,activation='softmax'))
model.summary()
#optimaztion
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X,y,batch_size=100,epochs=1,validation_split=0.33,verbose=1,)