import tensorflow as tf

def main():
    with tf.device('/cpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
        c = tf.matmul(a, b)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    print(sess.run(c))

main()
