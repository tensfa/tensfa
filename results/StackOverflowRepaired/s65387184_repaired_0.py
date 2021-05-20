from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
y_train_encoded =  to_categorical(Y_train)

input = keras.layers.Input(shape=(28,28))
hidden1 = keras.layers.Dense(128, activation="relu")(input)
hidden2 = keras.layers.Dense(128, activation="relu")(hidden1)
hidden3 = keras.layers.Dense(28, activation="relu")(hidden2)
output = keras.layers.Dense(10, activation="softmax")(hidden3)
model = keras.models.Model(inputs=[input], outputs=[output])

model.compile(loss="categorical_crossentropy",
             optimizer="adam",
             metrics=["accuracy"])

history=model.fit(X_train, y_train_encoded, epochs=1, validation_split=0.2)