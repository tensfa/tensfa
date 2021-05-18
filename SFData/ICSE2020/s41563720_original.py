def get_training_data(self):
    while 1:
        for i in range(1, 5):
            image = self.X_train[i]
            label = self.Y_train[i]
            yield (image, label)

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(32, 32, 3)))

history = model.fit_generator(get_training_data(),
                samples_per_epoch=1, nb_epoch=1,nb_val_samples=5,
                verbose=1,validation_data=get_validation_data())