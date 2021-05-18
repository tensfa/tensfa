import tensorflow as tf
import numpy

def global_contrast_normalize(X, scale=1., subtract_mean=True, use_std=False,
                              sqrt_bias=0., min_divisor=1e-8):
    mean = tf.reduce_mean(X, axis=1)
    if subtract_mean:
        X = X - mean[:, numpy.newaxis]  # Makes a copy.

    else:
        X = tf.copy.copy(X)
    if X.get_shape()[1] == 1:
        # ddof = 0
        mean, var = tf.nn.moments(X, axis=[1])

        normalizers = tf.sqrt(sqrt_bias + var) / scale

    else:
        normalizers = tf.sqrt(sqrt_bias + tf.reduce_sum((X ** 2), axis=1)) / scale
        Normalizers = tf.Variable(normalizers, 'float32')
        M = tf.Variable(min_divisor, 'float32')

    # normalizers = tf.Variable(tf.squeeze(normalizers))
    tf.cond(tf.squeeze(tf.less_equal(Normalizers, M)), lambda: tf.assign(Normalizers, [1]), lambda: tf.assign(Normalizers, normalizers))
    X /= Normalizers[:, tf.newaxis]  # Does not make a copy.
    return X

def main():
    x = tf.constant([[1,2]], dtype='float32')
    x = tf.Variable(x)
    global_contrast_normalize(x)

if __name__ == '__main__':
    main()

