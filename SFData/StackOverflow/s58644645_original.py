from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical

def load_dataset():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images.reshape((60000, 28 * 28))
    test_images = test_images.reshape((60000, 28 * 28))

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    return train_images, train_labels, test_images, test_labels

def prep_pixels(train, test):
    train_images = train_images.astype('float32') / 255
    test_images = test_images.astype('float32') / 255

    return train_images, test_images



def define_model():
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))

    return network

def compile(network):
    network.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

def run():
    (train_images, train_labels), (test_images, test_labels) = load_dataset()

    train_images, test_images = prep_pixels(test_images, test_images)

    network = define_model()

    compiled_network = compile(network)

    compiled_network.fit(train_images, train_labels, epochs=5, batch_size=128)

run()