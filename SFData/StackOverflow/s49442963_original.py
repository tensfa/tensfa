init_weights = he_normal()
main_input = Input(shape=(FEATURE_VECTOR_SIZE,)) #size 54
aux_input = Input(shape=(AUX_FEATURE_VECTOR_SIZE,)) #size 162
merged_input = concatenate([main_input, aux_input])

shared1 = Dense(164, activation='relu', kernel_initializer=init_weights)(merged_input)
shared2 = Dense(150, activation='relu', kernel_initializer=init_weights)(shared1)

main_output = Dense(NUM_ACTIONS, activation='linear', kernel_initializer=init_weights, name='main_output')(shared2)
aux_output = Dense(1, activation='linear', kernel_initializer=init_weights, name='aux_output')(shared2)

rms = RMSprop(lr=ALPHA)
model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer=rms, loss='mse')

aux_dummy = np.zeros(shape=(AUX_FEATURE_VECTOR_SIZE,))
print(aux_dummy.shape)
print(aux_dummy)
q_vals, _ = model.predict([encode_1_hot(next_state), aux_dummy], batch_size=1)