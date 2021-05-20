import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, Activation, Flatten,
                                     Conv2D, MaxPooling2D)
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
import numpy as np
import time

MODEL_NAME = f"_________{int(time.time())}"
BATCH_SIZE = 64

class ConvolutionalNetwork():
    '''
    A convolutional neural network to be used to classify images
    from the CIFAR-10 dataset.
    '''

    def __init__(self):
        '''
        self.training_images -- a 10000x3072 numpy array of uint8s. Each
                                a row of the array stores a 32x32 colour image.
                                The first 1024 entries contain the red channel
                                values, the next 1024 the green, and the final
                                1024 the blue. The image is stored in row-major
                                order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        self.training_labels -- a list of 10000 numbers in the range 0-9.
                                The number at index I indicates the label
                                of the ith image in the array data.
        '''
        # List of image categories

        self.training_images = np.random.randint(0, 256, (10000, 32*32*3), dtype=np.int32)
        self.training_labels = np.random.randint(0, 10, 10000, dtype=np.int32)

        # Reshaping the images + scaling
        self.shape_images()

        # Converts labels to one-hot
        self.training_labels = np.array(to_categorical(self.training_labels))

        self.create_model()

    def unpickle(self, file, encoding='bytes'):
        '''
        Unpickles the dataset files.
        '''
        with open(file, 'rb') as fo:
            training_dict = pickle.load(fo, encoding=encoding)
        return training_dict

    def shape_images(self):
        '''
        Reshapes the images and scales by 255.
        '''
        images = list()
        for d in self.training_images:
            image = np.zeros((32,32,3), dtype=np.uint8)
            image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
            image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
            image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
            images.append(image)

        for i in range(len(images)):
            images[i] = images[i]/255

        images = np.array(images)
        self.training_images = images
        print(self.training_images.shape)

    def create_model(self):
        '''
        Creating the ConvNet model.
        '''
        self.model = Sequential()
        self.model.add(Conv2D(64, (3, 3), input_shape=self.training_images.shape[1:]))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        self.model.add(Conv2D(64, (3,3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPooling2D(pool_size=(2,2)))

        # self.model.add(Flatten())
        # self.model.add(Dense(64))
        # self.model.add(Activation('relu'))

        self.model.add(Dense(10))
        self.model.add(Activation(activation='softmax'))

        self.model.compile(loss="categorical_crossentropy", optimizer="adam",
                           metrics=['accuracy'])

    def train(self):
        '''
        Fits the model.
        '''
        print(self.training_images.shape)
        print(self.training_labels.shape)
        self.model.fit(self.training_images, self.training_labels, batch_size=BATCH_SIZE,
                       validation_split=0.1, epochs=1)


network = ConvolutionalNetwork()
network.train()