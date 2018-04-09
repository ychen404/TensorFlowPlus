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
OUTPUT = 64
NUM_CLASSES = OUTPUT
#STR = "graph_fc_2_256.pb"
STR = "graph_conv_16_bat_4_" + str(OUTPUT) + ".pb"

with tf.Session() as sess:

    images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    with tf.name_scope('conv_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(images_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_1 = tf.nn.bias_add(conv, biases)
        print "The conv_1 shape is" + str (conv.get_shape())

#        shape = int(np.prod(out.get_shape()[1:]))
#        print "shape " + str(shape)
        #print "The shape is " + str (shape)
        #pool5_flat = tf.reshape(out[3:], [-1, shape])
        #print "The flat shape is " + str(pool5_flat)
#        out_0 = out[3:]
#        print "The shape of out_0 is " + str(out_0.get_shape())
#        conv1_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_2 = tf.nn.bias_add(conv, biases)
    
    with tf.name_scope('conv_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_3 = tf.nn.bias_add(conv, biases)
    
    with tf.name_scope('conv_4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_4 = tf.nn.bias_add(conv, biases)

    with tf.name_scope('conv_4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_4 = tf.nn.bias_add(conv, biases)
   
    with tf.name_scope('conv_5') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_5 = tf.nn.bias_add(conv, biases)

    with tf.name_scope('conv_6') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_6 = tf.nn.bias_add(conv, biases)
   
    with tf.name_scope('conv_7') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_6, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_7 = tf.nn.bias_add(conv, biases)


    with tf.name_scope('conv_8') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_7, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_8 = tf.nn.bias_add(conv, biases)

    with tf.name_scope('conv_9') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_9 = tf.nn.bias_add(conv, biases)

    with tf.name_scope('conv_10') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_9, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_10 = tf.nn.bias_add(conv, biases)

    with tf.name_scope('conv_11') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_10, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_11 = tf.nn.bias_add(conv, biases)


    with tf.name_scope('conv_12') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_11, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_12 = tf.nn.bias_add(conv, biases)
   
    with tf.name_scope('conv_13') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_12, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_13 = tf.nn.bias_add(conv, biases)
    
    with tf.name_scope('conv_14') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_13, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        conv_14 = tf.nn.bias_add(conv, biases)
    
    with tf.name_scope('conv_15') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, OUTPUT, OUTPUT], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_14, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[OUTPUT], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)

        shape = int(np.prod(out.get_shape()))
        flat = tf.reshape(out, [BATCH_SIZE, -1])
        out_0 = flat[:,0:OUTPUT]

    labels = tf.cast(labels_placeholder, tf.int32)
    oneHot = tf.one_hot (labels, NUM_CLASSES)



#    print "The shape of oneHot is " + str(oneHot.get_shape())
    loss = tf.reduce_mean (tf.square(tf.subtract(out_0, oneHot)), name='loss')
#    print loss
#    pdb.set_trace()
    #optimizer = tf.train.AdamOptimizer().minimize(loss, name = "train") 
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, name = "train") 
    init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
#   tf.train.write_graph (sess.graph_def, "models/", "graph_fc_2_128.pb", as_text=False)
    tf.train.write_graph (sess.graph_def, "/home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/assets/", STR, as_text=False)
