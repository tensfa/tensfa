from tensorflow.keras.layers import *
from tensorflow.keras import activations, losses, optimizers, models
import numpy as np

def get_base_model():
    inp = Input(shape=(48,))
    x = Reshape((-1,1))(inp)
    img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="valid")(x)
    img_1 = Convolution1D(16, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(32, kernel_size=2, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=2, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.01)(img_1)

    dense_1 = Dropout(0.01)(Dense(64, activation=activations.relu, name="dense_1")(img_1))

    base_model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)

    base_model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])

    return base_model

model = get_base_model()

X = np.random.random((335, 48))
y = np.random.random((335,))

model.fit(X, y, epochs=1)