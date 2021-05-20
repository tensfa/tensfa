from tensorflow import keras
import numpy as np

X_train = np.array(np.random.random((100, 4)), dtype=np.float32)
Y_train = np.random.randint(0, 4, 100, dtype=np.int32)
Y_train = np.eye(4)[Y_train]

X_test = np.array(np.random.random((50, 4)), dtype=np.float32)
Y_test = np.random.randint(0, 4, 50, dtype=np.int32)
Y_test = np.eye(4)[Y_test]

# define baseline model
def baseline_model():
    # create model
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(8, input_shape=(4, ), activation='relu'))
    model.add(keras.layers.Dense(4, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()

history = model.fit(X_train, Y_train, epochs=1)

test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)