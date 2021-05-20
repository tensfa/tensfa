import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

X = np.array(np.random.random((100, 2)), dtype=np.float)
y = np.random.randint(0,2,100,dtype=np.int32)
y = np.eye(2)[y]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(y_train.shape)

# Initialising the ANN
model = Sequential()
model.add(Dense(10, input_dim = 2, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

# Compiling the ANN
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, epochs = 1)