from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
import os

image_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1/255)

d = os.path.dirname(os.path.realpath(__file__))+'/../data/generator/train/'
training_generator = image_gen.flow_from_directory(d,target_size=(300,300), class_mode='binary')

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(300,300,3),activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(300,300),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(rate=0.35))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(training_generator,epochs=1)
