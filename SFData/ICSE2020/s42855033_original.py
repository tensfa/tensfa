first = Sequential()
first.add(Dense(1, input_shape=(2,), activation='sigmoid'))

second = Sequential()
second.add(Dense(1, input_shape=(1,), activation='sigmoid'))

result = Sequential()
merged = Concatenate([first, second])
ada_grad = Adagrad(lr=0.1, epsilon=1e-08, decay=0.0)
result.add(merged)
result.compile(optimizer=ada_grad, loss=_loss_tensor, metrics=['accuracy'])