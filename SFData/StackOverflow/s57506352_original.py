def Convolution(img):
    kernel = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
    sess = tf.Session()
    with tf.Session() as sess:
        img = img.astype('float32')
        Bias1 = tf.Variable(tf.truncated_normal(shape=[4], stddev=0.1))
        conv2d = tf.nn.conv2d(img, kernel, strides=[1, 1, 1, 1], padding='SAME')  # + Bias1
        conv2d = sess.run(conv2d)
    return conv2d

while True:
        with mss.mss() as sct:
                Game_Scr = np.array(sct.grab(Game_Scr_pos))[:,:,:3]

                cv2.imshow('Game_Src', Game_Scr)
                cv2.waitKey(0)

                Game_Scr = cv2.resize(Game_Scr, dsize=(960, 540), interpolation=cv2.INTER_AREA)
                print(Game_Scr.shape)

                print(Convolution(Game_Scr))