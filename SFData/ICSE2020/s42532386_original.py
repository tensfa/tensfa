import numpy
import matplotlib.pyplot as plt
import pandas
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# convert an array of values into a dataset matrix

def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 2])
    return numpy.array(dataX), numpy.array(dataY)



# fix random seed for reproducibility
numpy.random.seed(7)


# load the dataset
dataframe = pandas.read_csv('datos.csv', engine='python')
dataset = dataframe.values

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# reshape into X=t and Y=t+1
look_back = 3
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be  [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], look_back, 3))
testX = numpy.reshape(testX, (testX.shape[0],look_back, 3))

# create and fit the LSTM network

model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history= model.fit(trainX, trainY,validation_split=0.33, nb_epoch=200, batch_size=32)

# Plot training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('pérdida')
plt.xlabel('época')
plt.legend(['entrenamiento', 'validación'], loc='upper right')
plt.show()

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Get something which has as many features as dataset
trainPredict_extended = numpy.zeros((len(trainPredict),3))
# Put the predictions there
trainPredict_extended[:,2] = trainPredict[:,0]
# Inverse transform it and select the 3rd column.
trainPredict = scaler.inverse_transform(trainPredict_extended) [:,2]
print(trainPredict)
# Get something which has as many features as dataset
testPredict_extended = numpy.zeros((len(testPredict),3))
# Put the predictions there
testPredict_extended[:,2] = testPredict[:,0]
# Inverse transform it and select the 3rd column.
testPredict = scaler.inverse_transform(testPredict_extended)[:,2]


trainY_extended = numpy.zeros((len(trainY),3))
trainY_extended[:,2]=trainY
trainY=scaler.inverse_transform(trainY_extended)[:,2]


testY_extended = numpy.zeros((len(testY),3))
testY_extended[:,2]=testY
testY=scaler.inverse_transform(testY_extended)[:,2]


# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, 2] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, 2] = testPredict



#plot

 serie,=plt.plot(scaler.inverse_transform(dataset)[:,2])
prediccion_entrenamiento,=plt.plot(trainPredictPlot[:,2],linestyle='--')
prediccion_test,=plt.plot(testPredictPlot[:,2],linestyle='--')
plt.title('Consumo de agua')
plt.ylabel('cosumo (m3)')
plt.xlabel('dia')
plt.legend([serie,prediccion_entrenamiento,prediccion_test],['serie','entrenamiento','test'], loc='upper right')