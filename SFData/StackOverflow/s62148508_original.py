model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = 5,input_shape = (48,48,1)))
model.add(Conv2D(filters=64, kernel_size=5,strides=(1, 1), padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=1, padding='valid'))
model.add(Activation('relu'))

model.add(Conv2D(filters = 128, kernel_size = 5))
model.add(Conv2D(filters = 128, kernel_size=5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Conv2D(filters = 256, kernel_size = 5))
model.add(Conv2D(filters = 256, kernel_size=5))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(7,activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(input_array, output_array, batch_size = 64, epochs= 20, validation_split=0.10,)