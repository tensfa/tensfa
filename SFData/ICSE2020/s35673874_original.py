import tensorflow as tf
import numpy as np
import os
from PIL import Image
cur_dir = os.getcwd()

def modify_image(image):
  #resized = tf.image.resize_images(image, 180, 180, 3)
   image.set_shape([32,32,3])
   flipped_images = tf.image.flip_up_down(image)
   return flipped_images

def read_image(filename_queue):
  reader = tf.WholeFileReader()
  key,value = reader.read(filename_queue)
  image = tf.image.decode_jpeg(value)
  return key,image

def inputs():
 filenames = ['standard_1.jpg', 'standard_2.jpg' ]
 filename_queue = tf.train.string_input_producer(filenames)
 filename,read_input = read_image(filename_queue)
 reshaped_image = modify_image(read_input)
 reshaped_image = tf.cast(reshaped_image, tf.float32)
 label=tf.constant([1])
 return reshaped_image,label

def weight_variable(shape):
 initial = tf.truncated_normal(shape, stddev=0.1)
 return tf.Variable(initial)

def bias_variable(shape):
 initial = tf.constant(0.1, shape=shape)
 return tf.Variable(initial)

def conv2d(x, W):
 return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
 return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder(tf.float32, shape=[None,32,32,3])
y_ = tf.placeholder(tf.float32, shape=[None, 1])
image,label=inputs()
image=tf.reshape(image,[-1,32,32,3])
label=tf.reshape(label,[-1,1])
image_batch=tf.train.batch([image],batch_size=2)
label_batch=tf.train.batch([label],batch_size=2)

W_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])

image_4d=x_image = tf.reshape(image, [-1,32,32,3])

h_conv1 = tf.nn.relu(conv2d(image_4d, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([8 * 8 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy= -tf.reduce_sum(tf.cast(image_batch[1],tf.float32)*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(20000):
 sess.run(train_step,feed_dict={x:image_batch[0:1],y_:label_batch[0:1]})