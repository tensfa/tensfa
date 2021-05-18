textual_features = hashing_utility(filtered_words)  # Numpy array of hashed values(training data)

label_list = []  # Will eventually contain a list of Numpy arrays of binary one-hot labels

for index in range(one_hot_labels.shape[0]):
    label_list.append(one_hot_labels[index])

weighted_loss_value = (1 / (len(filtered_words)))  # Equal weight on each of the output layers' losses

weighted_loss_values = []

for index in range(one_hot_labels.shape[0]):
    weighted_loss_values.append(weighted_loss_value)

text_input = Input(shape=(1,))

intermediate_layer = Dense(64, activation='relu')(text_input)

hidden_bottleneck_layer = Dense(32, activation='relu')(intermediate_layer)

keras.regularizers.l2(0.1)

output_layers = []

for index in range(len(filtered_words)):
    output_layers.append(Dense(2, activation='sigmoid')(hidden_bottleneck_layer))

model = Model(inputs=text_input, outputs=output_layers)
model.compile(optimizer='RMSprop', loss='binary_crossentropy', metrics=['accuracy'], loss_weights=weighted_loss_values)

model.fit(textual_features, label_list, epochs=50)