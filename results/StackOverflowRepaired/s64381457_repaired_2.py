from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense
import numpy as np

embed_dim = 128
lstm_out = 200
batch_size = 32

X = np.array(np.random.random((100, 10)), dtype=np.float32)
Y = np.random.randint(0, 2, 100, dtype=np.int32)

model = Sequential()
model.add(Embedding(2500, embed_dim,input_length = X.shape[1]))
model.add(Dropout(0.2))
model.add(LSTM(lstm_out))
model.add(Dense(2,activation='sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Xtrain, Xtest, ytrain, ytest = train_test_split(X, Y, test_size = 0.2, shuffle=True)
ytrain = np.eye(2)[ytrain]
print(Xtrain.shape, ytrain.shape)
print(Xtest.shape, ytest.shape)

model.fit(Xtrain, ytrain, batch_size =batch_size, epochs = 1,  verbose = 5)