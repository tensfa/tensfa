# Encode label
le = LabelEncoder()
le.fit(glcm_df["label"].values)

print("categorical label : \n", le.classes_)

Y = le.transform(glcm_df['label'].values)
Y = to_categorical(Y)

# Split dataset
X_train, X_test, y_train, y_test = \
    train_test_split(X,
                     Y,
                     test_size=0.25,
                     random_state=42)


def nn_model(max_len):
    model = Sequential()
    model.add(Dense(32,
                    activation="elu",
                    input_shape=(max_len,)))
    model.add(Dense(1024, activation="elu"))
    model.add(Dense(512, activation="elu"))
    model.add(Dense(256, activation="elu"))
    model.add(Dense(128, activation="elu"))
    model.add(Dense(16))
    model.add(Activation("sigmoid"))

    model.summary()

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', precision, recall])

    return model


def check_model(model_, x, y, x_val, y_val, epochs_, batch_size_):
    hist = model_.fit(x,
                      y,
                      epochs=epochs_,
                      batch_size=batch_size_,
                      validation_data=(x_val, y_val))
    return hist


# Train data
max_len = X_train.shape[1]

EPOCHS = 1000
BATCH_SIZE = 32

model = nn_model(max_len)
history = check_model(model, X_train, y_train, X_test, y_test, EPOCHS, BATCH_SIZE)