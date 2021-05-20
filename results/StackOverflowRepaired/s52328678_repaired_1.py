import tensorflow as tf
import numpy as np

batch_size = 1
num_boxes = 500
num_classes = 3

boxes = np.array(np.random.random((batch_size, num_boxes, 4)), dtype=np.float32)
boxes = boxes.squeeze(0)
scores = np.array(np.random.random((batch_size, num_boxes)), dtype=np.float32)
scores = scores.squeeze(0)

selected_indices = tf.image.non_max_suppression(
    boxes=boxes,
    scores=scores,
    max_output_size=7,
    iou_threshold=0.5)