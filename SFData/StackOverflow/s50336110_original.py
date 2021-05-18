model =Sequential()
model.add(Dense(12, input_dim=3, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

x = np.array([[band1_input[input_cols_loop][input_rows_loop]],[band2_input[input_cols_loop][input_rows_loop]],[band3_input[input_cols_loop][input_rows_loop]]])

prediction_prob = model.predict(x)