model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(300,300,3),activation='relu'))
model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(300,300),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(300,300,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(rate=0.35))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(training_generator,epochs=5)