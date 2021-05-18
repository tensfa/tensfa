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

model.fit(X_train, y_train, epochs= 5, batch_size=32, verbose=True, validation_data=(X_test, y_test))