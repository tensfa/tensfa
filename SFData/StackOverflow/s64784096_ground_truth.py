import tensorflow.keras as kr
import numpy as np

#Create arrays
nn = [2, 4, 6, 8, 12, 10, 8, 2]
labels = []
words = []
docs_x = []
docs_y = []

INIT_LR = 1e-3
epochs = 6
batch_size = 64

training_data = np.array(np.random.random((100, 32)), dtype=np.float32)
target_data = np.random.randint(0, 2, 100, dtype=np.int32)
target_data = np.eye(2)[target_data]

model = kr.Sequential()

model.add(kr.layers.Dense(nn[1], activation='relu', input_shape=(32,)))
model.add(kr.layers.Dense(nn[2], activation='relu'))
model.add(kr.layers.Dense(nn[3], activation='relu'))
model.add(kr.layers.Dense(nn[4], activation='relu'))
model.add(kr.layers.Dense(nn[5], activation='relu'))
model.add(kr.layers.Dense(nn[6], activation='relu'))
model.add(kr.layers.Dense(nn[7], activation='sigmoid'))

print(model.summary())
model.compile(loss=kr.losses.categorical_crossentropy, optimizer=kr.optimizers.Adagrad(lr=INIT_LR, decay=INIT_LR / 100),metrics=['accuracy'])

#Training
model.fit(training_data , target_data , epochs=1)
