import cv2
import numpy as np
import tensorflow as tf
import os

def Convolution(img):
    kernel = tf.Variable(tf.truncated_normal(shape=[200, 200, 3, 3], stddev=0.1))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        img = img.astype('float32')
        conv2d = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')  # + Bias1
        conv2d = sess.run(conv2d)
    return conv2d


Game_Scr = cv2.imread(os.path.dirname(os.path.realpath(__file__))+"/../data/generator/train/class1/1.png")
Game_Scr = cv2.resize(Game_Scr, dsize=(960, 540), interpolation=cv2.INTER_AREA)
print(Game_Scr.shape)
print(Convolution(Game_Scr))