import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np
import pdb

#NUM_CLASSES = 64
#IMAGE_HEIGHT = 100 
IMAGE_HEIGHT = 32
#IMAGE_WIDTH = 100
IMAGE_WIDTH = 32
BATCH_SIZE = 96
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
OUTPUT = 10
NUM_CLASSES = 10
NUM_CONV = 5
#STR = "graph_fc_2_256.pb"
#STR = "graph_conv_1_bat_4_" + str(OUTPUT) + ".pb"
STR = "mobilenet_bat_" + str(BATCH_SIZE) + "_" + "conv_" + str(NUM_CONV) + ".pb"

with tf.Session() as sess:

    images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

    with tf.name_scope('conv_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32,
                                stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(images_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
                                 trainable=True, name='biases')
        out = tf.nn.bias_add(conv, biases)
        print "The conv shape is" + str (conv.get_shape())

        conv_1 = tf.layers.batch_normalization(out, training = True)
        conv_1 = tf.nn.relu(conv_1)
        print "The conv_1 shape is" + str (conv_1.get_shape())


    with tf.name_scope('conv_2_1_dw') as scope:
        print "conv_2_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,32,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_1,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())

        conv_2_1_dw = tf.nn.relu(out) 
        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_2_2') as scope:
        print "conv_2_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,32,64], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_2_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())
        conv_2_2 = tf.nn.relu(out)


    with tf.name_scope('conv_3_1_dw') as scope:
        print "conv_3_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,64,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())

        conv = tf.nn.depthwise_conv2d(conv_2_2,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())

        conv_3_1_dw = tf.nn.relu(out) 
        print "The shape of conv_dw_3_1 is" + str(conv_3_1_dw.get_shape())

    with tf.name_scope('conv_3_2') as scope:
        print "conv_3_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,64,128], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_3_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())
        conv_3_2 = tf.nn.relu(out)


    with tf.name_scope('conv_4_1_dw') as scope:
        print "conv_4_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,128,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_3_2,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())
        conv_4_1_dw = tf.nn.relu(out) 
        print "The shape of conv_4_1_dw is" + str(conv_4_1_dw.get_shape())


    with tf.name_scope('conv_4_2') as scope:
        print "conv_4_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,128,128], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_4_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())
        conv_4_2 = tf.nn.relu(out)


    with tf.name_scope('conv_5_1_dw') as scope:
        print "conv_5_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,128,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_4_2,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())
        conv_5_1_dw = tf.nn.relu(out) 
        print "The shape of conv_5_1_dw is" + str(conv_5_1_dw.get_shape())


    with tf.name_scope('conv_5_2') as scope:
        print "conv_5_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,128,256], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_5_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
        out = tf.layers.batch_normalization(conv, training = True)
        print "The shape of out is" + str(out.get_shape())
        conv_5_2 = tf.nn.relu(out)


   # with tf.name_scope('conv_6_1_dw') as scope:
   #     print "conv_6_1_dw"
   #     kernel = tf.Variable(tf.truncated_normal([3,3,256,1], dtype=tf.float32,stddev=1e-2), name='weights')
   #     print "The kernel shape is" + str (kernel.get_shape())
   #     conv = tf.nn.depthwise_conv2d(conv_5_2,kernel,strides=[1,1,1,1], padding='SAME')
   #     print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
   #     out = tf.layers.batch_normalization(conv, training = True)
   #     print "The shape of out is" + str(out.get_shape())
   #     conv_6_1_dw = tf.nn.relu(out) 
   #     print "The shape of conv_6_1_dw is" + str(conv_6_1_dw.get_shape())


   # with tf.name_scope('conv_6_2') as scope:
   #     print "conv_6_2"
   #     kernel = tf.Variable(tf.truncated_normal([1,1,256,256], dtype=tf.float32, stddev=1e-2), name='weights')
   #     conv = tf.nn.conv2d(conv_6_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
   #     print "The shape of the conv2d is " + str(conv.get_shape())
   #     out = tf.layers.batch_normalization(conv, training = True)
   #     print "The shape of out is" + str(out.get_shape())
   #     conv_6_2 = tf.nn.relu(out)


    # with tf.name_scope('conv_7_1_dw') as scope:
    #     print "conv_7_1_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,256,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_6_2, kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_7_1_dw = tf.nn.relu(out) 
    #     print "The shape of conv_dw_7_1 is" + str(conv_7_1_dw.get_shape())


    # with tf.name_scope('conv_7_2') as scope:
    #     print "conv_7_2"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,256,512], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_7_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_7_2 = tf.nn.relu(out)


    # with tf.name_scope('conv_8_1_dw') as scope:
    #     print "conv_8_1_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,512,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_7_2,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_1_dw = tf.nn.relu(out) 
    #     print "The shape of conv_8_1_dw is" + str(conv_8_1_dw.get_shape())


    # with tf.name_scope('conv_8_2') as scope:
    #     print "conv_8_2"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,512,512], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_8_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_2 = tf.nn.relu(out)


    # with tf.name_scope('conv_8_3_dw') as scope:
    #     print "conv_8_3_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,512,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_8_2,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_3_dw = tf.nn.relu(out) 
    #     print "The shape of conv_dw_8_3_dw is" + str(conv_8_3_dw.get_shape())


    # with tf.name_scope('conv_8_3') as scope:
    #     print "conv_8_3"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,512,512], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_8_3_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_3 = tf.nn.relu(out)


    # with tf.name_scope('conv_8_4_dw') as scope:
    #     print "conv_8_4_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,512,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_8_3,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_4_dw = tf.nn.relu(out) 


    # with tf.name_scope('conv_8_4') as scope:
    #     print "conv_8_4"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,512,512], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_8_4_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_4 = tf.nn.relu(out)

    # with tf.name_scope('conv_8_5_dw') as scope:
    #     print "conv_8_5_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,512,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_8_4,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_5_dw = tf.nn.relu(out) 

    # with tf.name_scope('conv_8_5') as scope:
    #     print "conv_8_5"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,512,512], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_8_5_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_8_5 = tf.nn.relu(out)

    # with tf.name_scope('conv_9_1_dw') as scope:
    #     print "conv_9_1_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,512,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_8_5,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_9_1_dw = tf.nn.relu(out) 

    # with tf.name_scope('conv_9_2') as scope:
    #     print "conv_9_2"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,512,1024], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_9_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_9_2 = tf.nn.relu(out)

    # with tf.name_scope('conv_10_1_dw') as scope:
    #     print "conv_10_1_dw"
    #     kernel = tf.Variable(tf.truncated_normal([3,3,1024,1], dtype=tf.float32,stddev=1e-2), name='weights')
    #     print "The kernel shape is" + str (kernel.get_shape())
    #     conv = tf.nn.depthwise_conv2d(conv_9_2,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the depthwise_conv is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_10_1_dw = tf.nn.relu(out) 

    # with tf.name_scope('conv_10_2') as scope:
    #     print "conv_10_2"
    #     kernel = tf.Variable(tf.truncated_normal([1,1,1024,1024], dtype=tf.float32, stddev=1e-2), name='weights')
    #     conv = tf.nn.conv2d(conv_10_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
    #     print "The shape of the conv is " + str(conv.get_shape())
    #     out = tf.layers.batch_normalization(conv, training = True)
    #     print "The shape of out is" + str(out.get_shape())
    #     conv_10_2 = tf.nn.relu(out)

    with tf.name_scope('avg_pool') as scope:
        print "avg_pool"
        avg_pool = tf.nn.avg_pool(conv_5_2, ksize=[1,7,7,1], strides=[1,1,1,1], padding='SAME')
        print "The shape of out is" + str(avg_pool.get_shape())

    with tf.name_scope('fc') as scope:
        print "fc"
        shape = int(np.prod(avg_pool.get_shape()[1:]))
        print "The shape is " + str(np.shape(shape))
        flat = tf.reshape(avg_pool, [-1, shape])
#       flat = tf.reshape(conv_dw_1, [4, -1])
        print "The shape is " + str (np.shape(shape))
        fc0w = tf.Variable(tf.truncated_normal([shape,OUTPUT],stddev=1e-2),name='weights')
        fc0b = tf.Variable(tf.constant(1.0, shape=[OUTPUT], dtype=tf.float32), trainable=True, name='biases')
        fc0l = tf.nn.bias_add(tf.matmul(flat, fc0w), fc0b)
#       print "shape " + str(shape)

    labels = tf.cast(labels_placeholder, tf.int32)
    oneHot = tf.one_hot (labels, NUM_CLASSES)
    print "The shape of oneHot is " + str(oneHot.get_shape())
    loss = tf.reduce_mean (tf.square(tf.subtract(fc0l, oneHot)), name='loss')
    print loss
#    pdb.set_trace()
    #optimizer = tf.train.AdamOptimizer().minimize(loss, name = "train") 
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, name = "train") 
    init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
#   tf.train.write_graph (sess.graph_def, "models/", "graph_fc_2_128.pb", as_text=False)
    tf.train.write_graph (sess.graph_def, "/home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/assets/", STR, as_text=False)
