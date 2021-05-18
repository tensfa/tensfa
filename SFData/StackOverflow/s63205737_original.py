# Neural Network.
import tensorflow as tf
print(tf.__version__)

import keras
print(keras.__version__)

from keras.models import Sequential
from keras.layers import Dense

# Create a new sequential model.
model = Sequential()

# Add input layer and Dense layer.
# Input layer contains 1 feature whereas first hidden layer has 5 neurons.
model.add(Dense(5,input_shape=(1,),activation="relu"))

# Add a final output one neuron layer.
model.add(Dense(1,activation="sigmoid"))

# Summarize a model:
model.summary()

# Model output shape.
print(model.output_shape)

# Model config.
print(model.get_config())

# List all weight tensors.
print(model.get_weights())

# Compile the Model.
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Fit the model.
model.fit(vector_array_train,y_train,epochs=20,batch_size=1, verbose=1)