batch_size = 32
num_epochs = 20

# Load MNIST dataset-
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Convert class vectors/target to binary class matrices or one-hot encoded values-
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

X_train.shape, y_train.shape
# ((60000, 28, 28), (60000, 10))

X_test.shape, y_test.shape


# ((10000, 28, 28), (10000, 10))


class LeNet300(Model):
    def __init__(self, **kwargs):
        super(LeNet300, self).__init__(**kwargs)

        self.flatten = Flatten()
        self.dense1 = Dense(units=300, activation='relu')
        self.dense2 = Dense(units=100, activation='relu')
        self.op = Dense(units=10, activation='softmax')

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.op(x)


# Instantiate an object using LeNet-300-100 dense model-
model = LeNet300()

# Compile the defined model-
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Define early stopping callback-
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=0.001,
    patience=3)

# Train defined and compiled model-
history = model.fit(
    x=X_train, y=y_train,
    batch_size=batch_size, shuffle=True,
    epochs=num_epochs,
    callbacks=[early_stopping_callback],
    validation_data=(X_test, y_test)
)