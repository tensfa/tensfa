import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def create_model(n_timesteps, n_features, n_outputs):
    tf.keras.backend.set_floatx('float16')
    dtype='float16'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(input_shape=(n_timesteps, n_features), kernel_size=3, filters=512),
        #tf.keras.layers.Dense(input_shape=(828,1), units=828, activation='relu', dtype=dtype),
        #tf.keras.layers.Dropout(0.2, dtype=dtype),
        tf.keras.layers.Dense(256, activation='relu', dtype=dtype),
        tf.keras.layers.Dropout(0.2, dtype=dtype),
        tf.keras.layers.Dense(n_outputs, activation='softmax', dtype=dtype)
    ])

    return model



if __name__ == "__main__":
    X_train = np.zeros((72148, 828, 1))
    y_train = np.zeros((72148, 23))
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
    model = create_model(n_timesteps, n_features, n_outputs)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    #model.compile(loss=loss_fn, optimizer=optimizer)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())



    batch_size = 32

    model.fit(X_train, y_train, epochs=3, batch_size= batch_size)