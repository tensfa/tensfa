import tensorflow as tf
import numpy as np

batch_size = 200

# this simulates a dataset read from a csv.....
x=np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]],dtype="float32")
y=np.array([0, 0, 0, 1],dtype="float32")

dataset = tf.data.Dataset.from_tensor_slices((x))
print(dataset)                  # <TensorSliceDataset shapes: (2,), types: tf.float32>
dataset = dataset.repeat(10000)
print('repeat ds ', dataset)    # repeat ds  <RepeatDataset shapes: (2,), types: tf.float32>

iter = dataset.make_initializable_iterator()
print('iterator ', iter)        # iterator  <tensorflow.python.data.ops.iterator_ops.Iterator object at 0x0000028589C62550>

sess = tf.Session()
sess.run(iter.initializer)
next_elt= iter.get_next()

print('shape of dataset ', dataset , '[iterator] elt ', next_elt)  # shape of dataset  <RepeatDataset shapes: (2,), types: tf.float32> [iterator] elt  Tensor("IteratorGetNext_105:0", shape=(2,), dtype=float32)
print('shape of it ', next_elt.shape) #s hape of it  (2,)
for i in range(4):
    print(sess.run(next_elt))
    ''' outputs: 
    [0. 0.]
    [1. 0.]
    [0. 1.]
    [1. 1.]

    '''

w = tf.Variable(tf.random_uniform([2,1], -1, 1, seed = 1234),name="weights_layer_1")
# this is where the error is because of shape mismatch of iterator and w variable.
# How od I make the shape of the iterator (2,1) so that matmul can be used?
# What is the proper way of aligning a tensor shape with inut data
# The output of the error:
#     ValueError: Shape must be rank 2 but is rank 1 for 'MatMul_19' (op: 'MatMul') with input shapes: [2], [2,1].
H = tf.matmul( sess.run(next_elt) , w)