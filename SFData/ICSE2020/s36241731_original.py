model.add(SimpleRNN(init='uniform',output_dim=1,input_dim=len(pred_frame.columns)))
model.compile(loss="mse", optimizer="sgd")
model.fit(X=predictor_train, y=target_train, batch_size=len(pred_frame.index),show_accuracy=True)