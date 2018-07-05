import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np

NUM_CLASSES = 102
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
BATCH_SIZE = 50
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001

with tf.Session() as sess:

    images_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped    = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])


    with tf.name_scope("conv1_1") as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64],dtype=tf.float32, stddev=1e-2),name="weights")
        conv = tf.nn.conv2d(images_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope("conv1_2") as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,64,64],dtype=tf.float32, stddev=1e-2),name="weights")
        conv = tf.nn.conv2d(conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv1_2 = tf.nn.relu(out, name=scope)

    pool1 = tf.nn.max_pool (conv1_2,ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool1')
        
    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d (conv1_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add (conv, biases)
        conv2_1 = tf.nn.relu (out, name=scope)

    with tf.name_scope('conv2_2') as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d (conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[128], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add (conv, biases)
        conv2_2 = tf.nn.relu (out, name=scope)

    pool2 = tf.nn.max_pool (conv2_2,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool2')

    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d (pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv3_2') as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d (conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv3_3') as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d (conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_3 = tf.nn.relu(out, name=scope)


    pool3 = tf.nn.max_pool (conv3_3,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')

    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv4_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_2 = tf.nn.relu(out, name=scope)
    
    with tf.name_scope('conv4_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_3 = tf.nn.relu(out, name=scope)
    
    pool4 = tf.nn.max_pool (conv4_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool4')   

    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)
    
    with tf.name_scope('conv5_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_2 = tf.nn.relu(out, name=scope)

    with tf.name_scope('conv5_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_3 = tf.nn.relu(out, name=scope)

    pool5 = tf.nn.max_pool (conv5_3,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='SAME',
                name='pool5')

    with tf.name_scope("fc6") as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        fc6w = tf.Variable(tf.truncated_normal([shape, 102],stddev=1e-2), name='weights')
        fc6b = tf.Variable(tf.constant(1.0, shape=[102],dtype=tf.float32),trainable=True, name='biases')
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
        fc6 = tf.nn.relu(fc6l)
        fc6 = tf.nn.dropout(fc6, 0.5)

    with tf.name_scope("fc7") as scope:
        fc7w = tf.Variable(tf.truncated_normal([102, 102],stddev=1e-2), name='weights')
        fc7b = tf.Variable(tf.constant(1.0, shape=[102],dtype=tf.float32),trainable=True, name='biases')
        fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
        fc7 = tf.nn.relu(fc7l)
        fc7 = tf.nn.dropout(fc7, 0.5)

    with tf.name_scope("fc8") as scope:
        fc8w = tf.Variable(tf.truncated_normal([102, 102],stddev=1e-2), name='weights')
        fc8b = tf.Variable(tf.constant(1.0, shape=[102],dtype=tf.float32),trainable=True, name='biases')
        fc8 = tf.nn.bias_add(tf.matmul(fc7, fc8w), fc8b, name='infer')

    labels = tf.cast(labels_placeholder, tf.int32)
    oneHot = tf.one_hot (labels, 102)
    loss = tf.reduce_mean (tf.square(tf.subtract(fc8, oneHot)), name='loss')
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, name = "train") 
    init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
    tf.train.write_graph (sess.graph_def, "models/", "graph_vgg16.pb", as_text=False)
