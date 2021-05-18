GmdMiss_Folder = os.path.join(os.getcwd(), '..', 'Photo', 'GMD Miss')
GmdMiss_List = os.listdir(GmdMiss_Folder)

for i in range(0, len(GmdMiss_List)):
    Img = os.path.join(os.getcwd(), GmdMiss_Folder, GmdMiss_List[i])
    Img = cv2.imread(Img, cv2.IMREAD_GRAYSCALE)
    Img = np.array(Img)
    Img = cv2.resize(Img, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
    Img_Miss_List.append(Img)
i=0

while True:
    Img = Img_Miss_List[i]
    with tf.Session() as sess:
        graph = tf.Graph()
        with graph.as_default():
            with tf.name_scope("Convolution"):
                Img = Convolution(Img)
i += 1

def Convolution(img):
    kernel = tf.Variable(tf.truncated_normal(shape=[180, 180, 3, 3], stddev=0.1))
    img = img.astype('float32')
    img = tf.nn.conv2d(np.expand_dims(img, 0), kernel, strides=[ 1, 15, 15, 1], padding='VALID')
    return img