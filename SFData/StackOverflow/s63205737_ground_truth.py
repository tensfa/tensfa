# Neural Network.
import tensorflow as tf
print(tf.__version__)

from tensorflow import keras
print(keras.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

train_n = 10
vector_array_train = np.array(np.random.random((train_n, 5000)),dtype=np.float32)
y_train = np.random.randint(0, 2, train_n, dtype=np.int32)

# Create a new sequential model.
model = Sequential()

# Add input layer and Dense layer.
# Input layer contains 1 feature whereas first hidden layer has 5 neurons.
model.add(Dense(5,input_shape=(5000,),activation="relu"))

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
model.fit(vector_array_train,y_train,epochs=1,batch_size=1, verbose=1)