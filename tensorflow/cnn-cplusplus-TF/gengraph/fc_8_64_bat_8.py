import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np

#NUM_CLASSES = 64
#IMAGE_HEIGHT = 100 
IMAGE_HEIGHT = 224
#IMAGE_WIDTH = 100
IMAGE_WIDTH = 224
BATCH_SIZE = 8
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
OUTPUT = 64
NUM_CLASSES = OUTPUT
#STR = "graph_fc_2_256.pb"
STR = "graph_fc_8_bat_8_" + str(OUTPUT) + ".pb"

with tf.Session() as sess:

    images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


    with tf.name_scope("fc0") as scope:
        shape = int(np.prod(images_reshaped.get_shape()[1:]))
        fc0w = tf.Variable(tf.truncated_normal([shape, OUTPUT],stddev=1e-2), name='weights')
        fc0b = tf.Variable(tf.constant(1.0, shape=[OUTPUT],dtype=tf.float32),trainable=True, name='biases')
        pool5_flat = tf.reshape(images_reshaped, [-1, shape])
        fc0l = tf.nn.bias_add(tf.matmul(pool5_flat, fc0w), fc0b)

    with tf.name_scope('fc1') as scope:
        fc1w = tf.Variable(tf.truncated_normal([OUTPUT, OUTPUT],stddev=1e-2), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[OUTPUT],dtype=tf.float32),trainable=True, name='biases')
        fc1l = tf.nn.bias_add(tf.matmul(fc0l, fc1w), fc1b)

    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable (tf.truncated_normal ([OUTPUT, OUTPUT],stddev=1e-2),name='weights')
        fc2b = tf.Variable (tf.constant (1.0, shape=[OUTPUT], dtype=tf.float32),
                    trainable=True, name='biases')
        fc2l = tf.nn.bias_add (tf.matmul (fc1l, fc2w), fc2b)

    with tf.name_scope('fc3') as scope:
        fc3w = tf.Variable (tf.truncated_normal ([OUTPUT, OUTPUT],stddev=1e-2),name='weights')
        fc3b = tf.Variable (tf.constant (1.0, shape=[OUTPUT], dtype=tf.float32),
                    trainable=True, name='biases')
        fc3l = tf.nn.bias_add (tf.matmul (fc2l, fc3w), fc3b)


    with tf.name_scope('fc4') as scope:
        fc4w = tf.Variable (tf.truncated_normal ([OUTPUT, OUTPUT],stddev=1e-2),name='weights')
        fc4b = tf.Variable (tf.constant (1.0, shape=[OUTPUT], dtype=tf.float32),
                    trainable=True, name='biases')
        fc4l = tf.nn.bias_add (tf.matmul (fc3l, fc4w), fc4b)


    with tf.name_scope('fc5') as scope:
        fc5w = tf.Variable (tf.truncated_normal ([OUTPUT, OUTPUT],stddev=1e-2),name='weights')
        fc5b = tf.Variable (tf.constant (1.0, shape=[OUTPUT], dtype=tf.float32),
                    trainable=True, name='biases')
        fc5l = tf.nn.bias_add (tf.matmul (fc4l, fc5w), fc5b)

    with tf.name_scope('fc6') as scope:
        fc6w = tf.Variable (tf.truncated_normal ([OUTPUT, OUTPUT],stddev=1e-2),name='weights')
        fc6b = tf.Variable (tf.constant (1.0, shape=[OUTPUT], dtype=tf.float32),
                    trainable=True, name='biases')
        fc6l = tf.nn.bias_add (tf.matmul (fc5l, fc6w), fc6b)

    with tf.name_scope('fc7') as scope:
        fc7w = tf.Variable (tf.truncated_normal ([OUTPUT, OUTPUT],stddev=1e-2),name='weights')
        fc7b = tf.Variable (tf.constant (1.0, shape=[OUTPUT], dtype=tf.float32),
                    trainable=True, name='biases')
        fc7l = tf.nn.bias_add (tf.matmul (fc6l, fc7w), fc7b)

    labels = tf.cast(labels_placeholder, tf.int32)
    oneHot = tf.one_hot (labels, NUM_CLASSES)
    loss = tf.reduce_mean (tf.square(tf.subtract(fc7l, oneHot)), name='loss')


    #optimizer = tf.train.AdamOptimizer().minimize(loss, name = "train") 
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, name = "train") 
    init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
#   tf.train.write_graph (sess.graph_def, "models/", "graph_fc_2_128.pb", as_text=False)
    tf.train.write_graph (sess.graph_def, "/home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/assets/", STR, as_text=False)
