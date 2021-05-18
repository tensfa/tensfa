import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Input, Flatten, Dense
import numpy as np

vocab_size = 1000
MAX_SEQUENCE_LENGTH = 100

embedding_matrix = np.array(np.random.random((vocab_size, 100)), dtype=np.float32)
X_train = np.array(np.random.random((64, 100)), dtype=np.float32)
y_train = np.random.randint(0, 5, 64, dtype=np.int32)
X_test = np.array(np.random.random((32, 100)), dtype=np.float32)
y_test = np.random.randint(0, 5, 32, dtype=np.int32)

y_train = tf.one_hot(y_train, 5)
y_test = tf.one_hot(y_test, 5)

# Add sequential model
model = Sequential()
# Add embedding layer
# No of output dimenstions is 100 as we embedded with Glove 100d
Embed_Layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=(MAX_SEQUENCE_LENGTH,), trainable=True)
#define Inputs
review_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype= 'int32', name = 'review_input')
review_embedding = Embed_Layer(review_input)
Flatten_Layer = Flatten()
review_flatten = Flatten_Layer(review_embedding)
output_size = 2

dense1 = Dense(100,activation='relu')(review_flatten)
dense2 = Dense(32,activation='relu')(dense1)
predict = Dense(2,activation='softmax')(dense2)

model = Model(inputs=[review_input],outputs=[predict])
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])
print(model.summary())

model.fit(X_train, y_train,
          steps_per_epoch=len(X_train)/32,
          epochs= 1, batch_size=32, verbose=True,
          validation_data=(X_test, y_test),
          validation_steps=len(X_test)/32)
