model = Sequential()

#C1
model.add(Convolution2D(15, 7, 7,
activation='relu',
border_mode='same',
input_shape=input_img_shape))
print("C1 shape: ", model.output_shape)

#S2
model.add(MaxPooling2D((2,2), border_mode='same'))
print("S2 shape: ", model.output_shape)
#...

#C5
model.add(Convolution2D(250, 5, 5,
activation='relu',
border_mode='same'))
print("C5 shape: ", model.output_shape)

#F6
model.add(Dense(50))