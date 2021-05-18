dataframe = pd.read_csv("iris.csv", header=None)
dataset = dataframe.values
X = dataset[:, 0:4].astype(float)
Y = dataset[:, 4]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
one_hot_y = keras.utils.to_categorical(encoded_Y)

X_train = X[:100]
X_test = X[50:]

Y_train = one_hot_y[:100]
Y_test = one_hot_y[50:]

print(X_train.shape)


# define baseline model
def baseline_model():
    # create model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, input_shape=(4, ), activation='relu'))
    model.add(keras.layers.Dense(3, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()

history = model.fit(X_train, X_test, epochs=5)

test_loss, test_acc = model.evaluate(Y_train, Y_test)
print('Test accuracy:', test_acc)