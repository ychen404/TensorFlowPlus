import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np
import pdb

#NUM_CLASSES = 64
#IMAGE_HEIGHT = 100 
IMAGE_HEIGHT = 224
#IMAGE_WIDTH = 100
IMAGE_WIDTH = 224
BATCH_SIZE = 4
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
OUTPUT = 2048
NUM_CLASSES = OUTPUT
#STR = "graph_fc_2_256.pb"
STR = "graph_conv_1_bat_4_" + str(OUTPUT) + ".pb"

with tf.Session() as sess:

    images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    with tf.name_scope('conv1_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(images_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
#        print "The conv shape is" + str (conv.get_shape())

        shape = int(np.prod(out.get_shape()))
#        print "shape " + str(shape)
        print "The shape is " + str (np.shape(shape))
        flat = tf.reshape(out, [4, -1])
        print "The flat shape is " + str(flat.shape)
        out_0 = flat[:,0:OUTPUT]
        print "The shape of out_0 is " + str(out_0.get_shape())
#        conv1_1 = tf.nn.relu(out, name=scope)

    
    labels = tf.cast(labels_placeholder, tf.int32)
    oneHot = tf.one_hot (labels, NUM_CLASSES)
    print "The shape of oneHot is " + str(oneHot.get_shape())
    loss = tf.reduce_mean (tf.square(tf.subtract(out_0, oneHot)), name='loss')
    print loss
#    pdb.set_trace()
    #optimizer = tf.train.AdamOptimizer().minimize(loss, name = "train") 
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, name = "train") 
    init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
#   tf.train.write_graph (sess.graph_def, "models/", "graph_fc_2_128.pb", as_text=False)
    tf.train.write_graph (sess.graph_def, "/home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/assets/", STR, as_text=False)
