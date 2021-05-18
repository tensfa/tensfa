sizeOfRow = len(data[0])
x = tensorFlow.placeholder("float", shape=[None, sizeOfRow])
y = tensorFlow.placeholder("float")

def neuralNetworkTrain(x):
  prediction = neuralNetworkModel(x)
  # using softmax function, normalize values to range(0,1)
  cost = tensorFlow.reduce_mean(tensorFlow.nn.softmax_cross_entropy_with_logits(prediction, y))

for temp in range(int(len(data) / batchSize)):
    ex, ey = takeNextBatch(i) # takes 500 examples
    i += 1
    # TO-DO : fix bug here
    temp, cos = sess.run([optimizer, cost], feed_dict= {x:ex, y:ey})