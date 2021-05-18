from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

textual_features = np.array(np.random.random((100, 1)), dtype=np.float32)
label_list = np.array(np.random.randint(0, 2, 100), dtype=np.int32)

text_input = Input(shape=(1,))
intermediate_layer = Dense(64, activation='relu')(text_input)
hidden_bottleneck_layer = Dense(32, activation='relu')(intermediate_layer)
keras.regularizers.l2(0.1)
output_layer = Dense(2, activation='sigmoid')(hidden_bottleneck_layer)
model = Model(inputs=text_input, outputs=output_layer)
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(textual_features, label_list, epochs=1)