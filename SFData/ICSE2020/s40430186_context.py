import numpy as np
import tensorflow as tf

# image parameters
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3
NUM_CLASSES = 2


def main():
    image = np.array(np.random.random((64,64,3)), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int32)
    with tf.Session() as sess:
        x_ = tf.placeholder(tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS])
        y_ = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])
        init = tf.global_variables_initializer()

        sess.run(init)
        my_classification = sess.run(tf.argmax(y_, 1), feed_dict={x_: image, y_:y})
        print('Neural Network predicted', my_classification[0], "for your image")

if __name__ == '__main__':
    main()