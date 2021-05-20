import numpy as np
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

X_train = np.array(np.random.random((100, 7, 1)), dtype=np.float32)

model = Sequential()
model.add(LSTM(150, activation='sigmoid', return_sequences=True, input_shape=X_train.shape[1:]))
model.add(Dropout(0.2))

