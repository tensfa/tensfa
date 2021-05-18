import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(40, input_shape=(21,), activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

arrTest1 = np.array([0.1,0.1,0.1,0.1,0.1,0.5,0.1,0.0,0.1,0.6,0.1,0.1,0.0,0.0,0.0,0.1,0.0,0.0,0.1,0.0,0.0])
scores = model.predict(arrTest1)
print(scores)