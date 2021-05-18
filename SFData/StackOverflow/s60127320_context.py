import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

k = 7
feature_num = 9
batch=1
no_bins = k*100 if k*100 < 1000 else 1000
train_num = 100
test_num = 10

train = np.array(np.random.random((train_num, feature_num)), dtype=np.float32)
test = np.array(np.random.random((test_num, feature_num)), dtype=np.float32)
redshift = np.random.randint(0, 1000, train_num, dtype=np.int32)

max_z = np.max(redshift)
min_z = np.min(redshift)

model = Sequential()
model.add(Dense(feature_num, input_dim=feature_num, kernel_initializer='normal', use_bias=True, activation='relu'))
model.add(Dense(1, kernel_initializer='normal', use_bias=True))
model.compile(loss='mean_squared_error', optimizer='adam')

edges = np.histogram(redshift[::batch], bins=no_bins, range=(min_z,max_z))[1]
edges_with_overflow = np.histogram(redshift[::batch], bins=no_bins+1, range=(min_z, max_z))[1]
model.fit(train[::batch], edges_with_overflow[np.digitize(redshift[::batch], edges)], epochs=1)
prediction = []
for point in test:
    prediction.append(model.predict([point])[0])
