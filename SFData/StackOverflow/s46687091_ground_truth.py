import tensorflow as tf
import numpy as np
import math

batch_size = 4
embedding_size = 7

vocabulary_size = 10
emb_size = 32

glove_embeddings_arr = np.random.random((vocabulary_size, emb_size))
glove_embeddings_arr = np.array(glove_embeddings_arr, dtype=np.float32)
num_sampled = 1

input_data = tf.placeholder(tf.int32, shape=[batch_size, embedding_size])
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, emb_size], stddev=1.0 / math.sqrt(embedding_size)))

nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
embed = tf.nn.embedding_lookup(tf.convert_to_tensor(glove_embeddings_arr), input_data)

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases, labels=labels, inputs=tf.reduce_sum(embed, 1), num_sampled=num_sampled, num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)