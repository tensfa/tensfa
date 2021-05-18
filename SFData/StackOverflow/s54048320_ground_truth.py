import os

import cv2
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# path to the training, validation, and testing directories
train_directory = os.path.dirname(os.path.realpath(__file__))+"/../data/generator/train/"
validation_directory = os.path.dirname(os.path.realpath(__file__))+"/../data/generator/valid/"
test_directory = os.path.dirname(os.path.realpath(__file__))+"/../data/generator/test/"
results_directory = os.path.dirname(os.path.realpath(__file__))+"/../data/generator/results/"
number_of_training_samples = 6
number_of_validation_samples = 2
number_of_test_samples = 2
batch_size = 1

# get back the convolutional part of a VGG network trained on ImageNet
conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(512,512,3))
conv_base.summary()

# preprocess the data
# rescale images by the factor 1/255
train_data = ImageDataGenerator(rescale=1.0/255)
validation_data = ImageDataGenerator(rescale=1.0/255)
test_data = ImageDataGenerator(rescale=1.0/255)

train_features = np.zeros(shape=(number_of_training_samples,16,16,512))
train_labels = np.zeros(shape=(number_of_training_samples))

train_generator = train_data.flow_from_directory(
    train_directory,
    target_size=(512,512),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True)

i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = conv_base.predict(inputs_batch)
    train_features[i*batch_size:(i+1)*batch_size] = features_batch
    train_labels[i*batch_size:(i+1)*batch_size] = labels_batch
    i += 1
    if i * batch_size >= number_of_training_samples:
        break

train_features = np.reshape(train_features, (number_of_training_samples,16*16*512))

validation_features = np.zeros(shape=(number_of_validation_samples,16,16,512))
validation_labels = np.zeros(shape=(number_of_validation_samples))

validation_generator = validation_data.flow_from_directory(
    validation_directory,
    target_size=(512,512),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = conv_base.predict(inputs_batch)
    validation_features[i*batch_size:(i+1)*batch_size] = features_batch
    validation_labels[i*batch_size:(i+1)*batch_size] = labels_batch
    i += 1
    if i * batch_size >= number_of_validation_samples:
        break

validation_features = np.reshape(validation_features, (number_of_validation_samples,16*16*512))

test_generator = test_data.flow_from_directory(
    test_directory,
    target_size=(512,512),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

# define the Convolutional Neural Network (CNN) model
model = models.Sequential()
model.add(layers.Dense(1024,activation='relu',input_dim=16*16*512))
model.add(layers.Dense(1,activation='sigmoid'))

# compile the model

model.compile(loss='binary_crossentropy',
    optimizer=optimizers.Adam(lr=0.01),
    metrics=['acc'])

# fit the model to the data
history = model.fit(train_features,
    train_labels,
    epochs=1,
    batch_size=batch_size,
    validation_data=(validation_features,validation_labels))

# predict classes
for root, dirs, files in os.walk(test_directory):
    for file in files:
        img = cv2.imread(root + '/' + file)
        img = cv2.resize(img,(512,512),interpolation=cv2.INTER_AREA)
        img = np.expand_dims(img, axis=0)
        img = img/255.0
        feature_value = conv_base.predict(img)
        feature_value= np.reshape(feature_value,(1,16*16*512))
        img_class = model.predict_classes(feature_value)
        prediction = img_class[0]