import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import Sequential
import os
# Define Window size (color images)
img_window = (10,10,3)

# Flatten the Window shape
input_shape = np.prod(img_window)
print(input_shape)

# Define MLP with two hidden layers(neurons)
simpleMLP = Sequential(
    [layers.Input(shape=img_window),
     layers.Flatten(), # Flattens the input, conv2D to 1 vector , which does not affect the batch size.
     layers.Dense(input_shape//2 ,activation="relu"),
     layers.Dense(input_shape//2 ,activation="relu"),
     layers.Dense(2,activation="sigmoid"),
     ]
)
# After model is "built" call its summary() menthod to display its contents
simpleMLP.summary()

# Initialization
# Size of the batches of data, adjust it depends on RAM
batch_size = 1
epochs = 1
# Compile MLP model with 3 arguments: loss function, optimizer, and metrics function to judge model performance
simpleMLP.compile(loss="binary_crossentropy",optimizer="adam",metrics=["binary_accuracy"])  #BCE

# Create ImagedataGenerator to splite training, validation dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = os.path.dirname(os.path.realpath(__file__))+'/../data/generator/train'
train_datagen = ImageDataGenerator(
    rescale=1./255, # rescaling factor
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

valid_dir = os.path.dirname(os.path.realpath(__file__))+'/../data/generator/valid'
valid_datagen =ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_window[:2],
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb'
    )

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=img_window[:2],
    batch_size=batch_size,
    class_mode='binary',
    color_mode='rgb')

# Train the MLP model
simpleMLP.fit_generator(
    train_generator,
    steps_per_epoch= 3 // batch_size,
    epochs=1,
    validation_data=validation_generator,
    validation_steps= 1 // batch_size)
