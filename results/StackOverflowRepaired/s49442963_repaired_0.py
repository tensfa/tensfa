from tensorflow.keras.initializers import he_normal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf

import numpy as np

FEATURE_VECTOR_SIZE = 54
AUX_FEATURE_VECTOR_SIZE = 162
NUM_ACTIONS = 3
ALPHA = 0.001

init_weights = he_normal()
main_input = Input(shape=(FEATURE_VECTOR_SIZE,)) #size 54
aux_input = Input(shape=(AUX_FEATURE_VECTOR_SIZE,)) #size 162
merged_input = tf.concat([main_input, aux_input], -1)

shared1 = Dense(164, activation='relu', kernel_initializer=init_weights)(merged_input)
shared2 = Dense(150, activation='relu', kernel_initializer=init_weights)(shared1)

main_output = Dense(NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(shared2)
aux_output = Dense(1, activation='linear', kernel_initializer=init_weights, name='aux_output')(shared2)

rms = RMSprop(lr=ALPHA)
model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer=rms, loss='mse')

aux_dummy = np.zeros(shape=(AUX_FEATURE_VECTOR_SIZE,))
encode_1_hot = np.zeros(shape=(1, FEATURE_VECTOR_SIZE))
q_vals, _ = model.predict([encode_1_hot, aux_dummy], batch_size=1)