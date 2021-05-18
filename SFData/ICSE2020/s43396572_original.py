import numpy
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
numpy.random.seed(7)
datasetTraining = numpy.loadtxt("CancerAdapter.csv",delimiter=",")
X = datasetTraining[:,1:31]
Y = datasetTraining[:,0]
datasetTesting = numpy.loadtxt("CancereEvaluation.csv",delimiter=",")
X_test = datasetTraining[:,1:31]
Y_test = datasetTraining[:,0]
model = Sequential()
model.add(Conv1D(2,2,activation='relu',input_shape=X.shape))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=5)
scores = model.evaluate(X_test, Y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))