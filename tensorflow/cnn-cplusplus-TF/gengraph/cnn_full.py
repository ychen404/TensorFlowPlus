import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np

NUM_CLASSES = 10
IMAGE_HEIGHT = 32 
IMAGE_WIDTH = 32
BATCH_SIZE = 128
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001

with tf.Session() as sess:

    #images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), name="input")
    images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    with tf.name_scope("conv1_1") as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 3, 64],dtype=tf.float32, stddev=1e-2),name="weights")
        #conv = tf.nn.conv2d (images_placeholder,kernel, [1, 1, 1, 1], padding='SAME')
        conv = tf.nn.conv2d (images_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[64], dtype=tf.float32),
                                trainable=True, name='biases')
        out = tf.nn.bias_add (conv, biases)
        conv1_1 = tf.nn.relu (out, name=scope)

    pool1 = tf.nn.max_pool (conv1_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
        

    with tf.name_scope('conv2_1') as scope:
        kernel = tf.Variable (tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, 
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d (pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable (tf.constant(0.0, shape=[128], dtype=tf.float32),
                                trainable=True, name='biases')
        out = tf.nn.bias_add (conv, biases)
        conv2_1 = tf.nn.relu (out, name=scope)

    pool2 = tf.nn.max_pool (conv2_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool2')

    with tf.name_scope('conv3_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
                                    stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                                    trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv3_1 = tf.nn.relu(out, name=scope)

    pool3 = tf.nn.max_pool (conv3_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool3')

    with tf.name_scope('conv4_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv4_1 = tf.nn.relu(out, name=scope)

    pool4 = tf.nn.max_pool (conv4_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool4')   

    with tf.name_scope('conv5_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        conv5_1 = tf.nn.relu(out, name=scope)

    pool5 = tf.nn.max_pool (conv5_1,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME',
                            name='pool5')
    print "The shape of pool5 is " + str(pool5.get_shape())

    with tf.name_scope("fc0") as scope:
        #fc0w = tf.Variable(tf.truncated_normal([IMAGE_WIDTH*IMAGE_HEIGHT, 4096],stddev=1e-2), name='weights')
        shape = int(np.prod(pool5.get_shape()[1:]))
        print shape
                
        fc0w = tf.Variable(tf.truncated_normal([shape, 4096],stddev=1e-2), name='weights')
        print "fc0w " + str(fc0w.get_shape())
        fc0b = tf.Variable(tf.constant(1.0, shape=[4096],dtype=tf.float32),trainable=True, name='biases')
        print "fc0b " + str(fc0b.get_shape())
        pool5_flat = tf.reshape(pool5, [-1, shape])
        print "pool5_flat " + str(pool5_flat.get_shape())
        #fc0l = tf.nn.bias_add(tf.matmul(images_placeholder, fc0w), fc0b)
        fc0l = tf.nn.bias_add(tf.matmul(pool5_flat, fc0w), fc0b)
        print "fc0l " + str(fc0l.get_shape())
        fc0 = tf.nn.relu(fc0l)

    with tf.name_scope('fc1') as scope:
                #shape = 802816
        #shape = int(np.prod(pool1.get_shape()[1:]))
        fc1w = tf.Variable(tf.truncated_normal([4096, 4096],stddev=1e-2), name='weights')
        fc1b = tf.Variable(tf.constant(1.0, shape=[4096],dtype=tf.float32),trainable=True, name='biases')
        #pool1_flat = tf.reshape(pool1, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(fc0, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        #fc1 = tf.nn.dropout(fc1, 0.5)


    with tf.name_scope('fc2') as scope:
        fc2w = tf.Variable (tf.truncated_normal ([4096, NUM_CLASSES],stddev=1e-2),name='weights')
        fc2b = tf.Variable (tf.constant (1.0, shape=[NUM_CLASSES], dtype=tf.float32),
                    trainable=True, name='biases')
        #pool1_flat = tf.reshape(pool1, [-1, shape])
              #  fc2l = tf.nn.bias_add (tf.matmul (fc1, fc2w), fc2b)
        fc2l = tf.nn.bias_add (tf.matmul (fc1, fc2w), fc2b, name = 'infer')

              #  fc3l = tf.nn.bias_add (tf.matmul (pool1_flat, fc2w), fc2b)


    labels = tf.cast(labels_placeholder, tf.int32)
    print ("The shape of labels is " + str(labels.get_shape()))
    oneHot = tf.one_hot (labels, NUM_CLASSES)
    print ("The shape of oneHot is " + str(oneHot.get_shape()))
    print ("The shape of fc2l is " + str (fc2l.get_shape()))
#        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits (labels=labels,
#                 logits=fc3l, name="xentropy")
#    loss = tf.reduce_mean (tf.square(tf.subtract(fc2l, oneHot)), name='loss')

#    loss = tf.losses.softmax_cross_entropy(oneHot, fc2l, scope='loss')

    loss = tf.nn.softmax_cross_entropy_with_logits (labels = oneHot, logits = fc2l, name='loss')
#       print loss
        #cost = tf.reduce_sum (tf.square (fc3l - labels), name="cost")

#   optimizer = tf.train.AdamOptimizer (LEARNING_RATE)
#   global_step = tf.Variable (0, name='global_step', trainable=False)
#   train_op = optimizer.minimize (loss, global_step=global_step, name="train")

    #optimizer = tf.train.AdamOptimizer().minimize(loss, name = "train") 
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss, name = "train") 
    init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
    tf.train.write_graph (sess.graph_def, "/home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/assets", "graph_full_bat_128_softmax.pb", as_text=False)
