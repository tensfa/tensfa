def baseline_model():
    model = models.Sequential()
    model.add(layers.Conv1D(1, 5, input_shape=(6,1), activation="tanh"))
    model.add(layers.MaxPool1D(pool_size=2))
    model.add(layers.core.Flatten())
    model.add(layers.Dense(2))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model

# df is pandas DataFrame
X = np.array(df[['rp', 'x', 'y', 'class', 'at', 'dt']], dtype=np.float64)
y = np.array(df[['ap', 'dp']], dtype=np.float64)
# X = np.expand_dims(X, -1)
# y = np.expand_dims(y, -1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
mode = baseline_model()
history = mode.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_test, y_test))

X=np.array([[-69.3078,   0.    ,   1.    ,   1.    ,  90.    ,  90.    ],
       [-69.4585,   0.    ,   2.    ,   1.    ,  90.    ,  90.    ],
       [-69.4776,   0.    ,   3.    ,   1.    ,  90.    ,  90.    ],
       ...,
       [-65.8291,  35.    ,  33.    ,   1.    ,  90.    ,  90.    ],
       [-71.0137,  35.    ,  34.    ,   1.    ,  90.    ,  90.    ],
       [-67.2308,  35.    ,  35.    ,   1.    ,  90.    ,  90.    ]])
y=np.array([[ 15.4463, -17.5046],
       [ 15.4777, -17.536 ],
       [ 15.5092, -17.5675],
       ...,
       [ 15.8361, -17.8944],
       [ 15.8809, -17.9392],
       [ 15.9259, -17.9842]])

# X,y type is numpy array
# X shape is (4725, 6) ,y shape is (4725, 2)
# X[0] shape is (6,) , y[0] shape is (2,)