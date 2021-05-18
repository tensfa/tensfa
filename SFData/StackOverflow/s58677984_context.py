import os
import numpy as np
import tensorflow as tf
import cv2

def Convolution(img):
    kernel = tf.Variable(tf.truncated_normal(shape=[180, 180, 3, 3], stddev=0.1))
    img = img.astype('float32')
    img = tf.nn.conv2d(np.expand_dims(img, 0), kernel, strides=[ 1, 15, 15, 1], padding='VALID')
    return img

GmdMiss_Folder = os.path.dirname(os.path.realpath(__file__))+ '/../data/generator/train/class1'
GmdMiss_List = os.listdir(GmdMiss_Folder)

Img_Miss_List = []
for i in range(0, len(GmdMiss_List)):
    Img = os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[i])
    Img = cv2.imread(Img, cv2.IMREAD_GRAYSCALE)
    Img = cv2.resize(Img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
    Img_Miss_List.append(Img)

for Img in Img_Miss_List:
    with tf.Session() as sess:
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("Convolution"):
                Img = Convolution(Img)