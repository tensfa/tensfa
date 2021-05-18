loaded_graph = tf.Graph()

image_to_use = train_images[0]
print(image_to_use.shape) # (128, 128, 3)

with tf.Session(graph=loaded_graph) as sess:

    # Load model
    loader = tf.train.import_meta_graph(save_model_path + ".meta")
    loader.restore(sess, save_model_path)

    # Get Tensors from loaded model
    hidden_layer_1 = loaded_graph.get_tensor_by_name("hidden-layer-1:0")
    keep_prob_tf = tf.placeholder(tf.float32, name="keep-prob-in")
    image_in_tf = tf.placeholder(tf.float32, [None, image_to_use.shape[0], image_to_use.shape[1], image_to_use.shape[2]], name="image-in")

    units = sess.run(hidden_layer_1, feed_dict={image_in_tf:image_to_use, keep_prob_tf:1.0})