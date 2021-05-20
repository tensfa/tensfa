import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

img_data = np.array(np.random.random((376, 128, 128, 3)), dtype=np.float32)
num_of_samples = img_data.shape[0]
input_shape = img_data[0].shape

X_train, X_test = train_test_split(img_data, test_size=0.2, random_state=2)

inputTensor = Input(input_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputTensor)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded_data = MaxPooling2D((2, 2), padding='same')(x)

encoder_model = Model(inputTensor,encoded_data)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional
encoded_input = Input((4,4,8))
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_input)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded_data = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder_model = Model(encoded_input, decoded_data)

autoencoder_input = Input(input_shape)
encoded = encoder_model(autoencoder_input)
decoded = decoder_model(encoded)
autoencoder_model = Model(autoencoder_input, decoded)
autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder_model.fit(X_train, X_train,
            epochs=1,
            batch_size=32,
            validation_data=(X_test, X_test))
