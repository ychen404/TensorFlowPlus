import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np
import pdb

IMAGE_HEIGHT = 32
#IMAGE_WIDTH = 100
IMAGE_WIDTH = 32
BATCH_SIZE = 64
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001
OUTPUT = 64
NUM_CLASSES = OUTPUT
#STR = "graph_fc_2_256.pb"
#STR = "graph_conv_1_bat_4_" + str(OUTPUT) + ".pb"
#STR = "mobilenet_simplified_" + str(OUTPUT) + ".pb"
STR = "graph_conv_depth_8_bat_" + str(BATCH_SIZE) + "_" + "w_" + str(OUTPUT) + ".pb"

with tf.Session() as sess:

    images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
    labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
    images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

 #   with tf.name_scope('conv_1') as scope:
 #       kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 32], dtype=tf.float32,
 #                               stddev=1e-2), name='weights')
 #       conv = tf.nn.conv2d(images_reshaped, kernel, [1, 2, 2, 1], padding='SAME')
 #       biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32),
 #                                trainable=True, name='biases')
 #       conv_1 = tf.nn.bias_add(conv, biases)
 #       print "The conv shape is" + str (conv.get_shape())
 #
 #       #conv_1 = tf.layers.batch_normalization(out, training = True)
 #       #conv_1 = tf.nn.relu(out)
 #       print "The conv_1 shape is" + str (conv_1.get_shape())


    with tf.name_scope('conv_1_1_dw') as scope:
        print "conv_1_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,3,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(images_reshaped,kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_1_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_1_2') as scope:
        print "conv_1_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,3,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_1_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_1_2 = conv
        print "The shape of conv_1_2 is " + str(conv_1_2.get_shape())
     #   flat = tf.reshape(conv_1_2, [BATCH_SIZE, -1])
     #   print "The shape of flat is " + str(flat.get_shape())
     #   out_0 = flat[:,0:OUTPUT]



    with tf.name_scope('conv_2_1_dw') as scope:
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv_2_1_dw = tf.nn.depthwise_conv2d(conv_1_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
#        conv_1_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_2_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_2_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_2_2 = conv
  #      print "The shape of conv_1_2 is " + str(conv_1_2.get_shape())
   #     flat = tf.reshape(conv_1_2, [BATCH_SIZE, -1])
    #    print "The shape of flat is " + str(flat.get_shape())
     #   out_0 = flat[:,0:OUTPUT]


    with tf.name_scope('conv_3_1_dw') as scope:
        print "conv_3_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_2_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_3_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_3_2') as scope:
        print "conv_3_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_3_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_3_2 = conv
        print "The shape of conv_1_2 is " + str(conv_1_2.get_shape())
      #  flat = tf.reshape(conv_1_2, [BATCH_SIZE, -1])
      #  print "The shape of flat is " + str(flat.get_shape())
      #  out_0 = flat[:,0:OUTPUT]


    with tf.name_scope('conv_4_1_dw') as scope:
        print "conv_4_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_3_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_4_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_4_2') as scope:
        print "conv_4_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_4_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_4_2 = conv
        # print "The shape of conv_4_2 is " + str(conv_4_2.get_shape())
        # flat = tf.reshape(conv_4_2, [BATCH_SIZE, -1])
        # print "The shape of flat is " + str(flat.get_shape())
        # out_0 = flat[:,0:OUTPUT]



    with tf.name_scope('conv_5_1_dw') as scope:
        print "conv_5_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_4_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_5_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_5_2') as scope:
        print "conv_5_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_5_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_5_2 = conv
        print "The shape of conv_5_2 is " + str(conv_4_2.get_shape())
        # flat = tf.reshape(conv_4_2, [BATCH_SIZE, -1])
        # print "The shape of flat is " + str(flat.get_shape())
        # out_0 = flat[:,0:OUTPUT]

    with tf.name_scope('conv_6_1_dw') as scope:
        print "conv_6_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_5_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_6_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_6_2') as scope:
        print "conv_6_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_6_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_6_2 = conv
        #print "The shape of conv_6_2 is " + str(conv_6_2.get_shape())
        # flat = tf.reshape(conv_4_2, [BATCH_SIZE, -1])
        # print "The shape of flat is " + str(flat.get_shape())
        # out_0 = flat[:,0:OUTPUT]

    with tf.name_scope('conv_7_1_dw') as scope:
        print "conv_7_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_6_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_7_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_7_2') as scope:
        print "conv_7_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_7_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_7_2 = conv
        print "The shape of conv_7_2 is " + str(conv_4_2.get_shape())
        flat = tf.reshape(conv_7_2, [BATCH_SIZE, -1])
        print "The shape of flat is " + str(flat.get_shape())
        out_0 = flat[:,0:OUTPUT]

    with tf.name_scope('conv_8_1_dw') as scope:
        print "conv_8_1_dw"
        kernel = tf.Variable(tf.truncated_normal([3,3,OUTPUT,1], dtype=tf.float32,stddev=1e-2), name='weights')
        print "The kernel shape is" + str (kernel.get_shape())
        conv = tf.nn.depthwise_conv2d(conv_7_2, kernel,strides=[1,1,1,1], padding='SAME')
        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
       # out = tf.layers.batch_normalization(conv, training = True)
       # print "The shape of out is" + str(out.get_shape())
#        conv_2_1_dw = tf.nn.relu(out) 
        conv_8_1_dw = conv
#        print "The shape of conv_2_1_dw is" + str(conv_2_1_dw.get_shape())

    with tf.name_scope('conv_8_2') as scope:
        print "conv_8_2"
        kernel = tf.Variable(tf.truncated_normal([1,1,OUTPUT,OUTPUT], dtype=tf.float32, stddev=1e-2), name='weights')
        conv = tf.nn.conv2d(conv_8_1_dw,kernel,strides=[1,1,1,1], padding='SAME')
#        print "The shape of the depthwise_conv2d is " + str(conv.get_shape())
#        out = tf.layers.batch_normalization(conv, training = True)
#        print "The shape of out is" + str(out.get_shape())
#        conv_2_2 = tf.nn.relu(out)
        conv_8_2 = conv
        print "The shape of conv_8_2 is " + str(conv_4_2.get_shape())
        flat = tf.reshape(conv_8_2, [BATCH_SIZE, -1])
        print "The shape of flat is " + str(flat.get_shape())
        out_0 = flat[:,0:OUTPUT]


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