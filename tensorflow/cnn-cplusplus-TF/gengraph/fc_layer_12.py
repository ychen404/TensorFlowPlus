import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

import random
import numpy as np

NUM_CLASSES = 102
#IMAGE_HEIGHT = 100 
IMAGE_HEIGHT = 224
#IMAGE_WIDTH = 100
IMAGE_WIDTH = 224
BATCH_SIZE = 50
NUM_CHANNELS = 3
LEARNING_RATE = 0.0001

with tf.Session() as sess:

	#images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS), name="input")
	images_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE, IMAGE_HEIGHT * IMAGE_WIDTH * NUM_CHANNELS), name="input")
	labels_placeholder = tf.placeholder (tf.float32, shape=(BATCH_SIZE), name="label")
        images_reshaped = tf.reshape(images_placeholder, [BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])

#	with tf.name_scope("conv1_1") as scope:
#                kernel = tf.Variable (tf.truncated_normal([3, 3, 3, 64],dtype=tf.float32, stddev=1e-2),name="weights")
#		#conv = tf.nn.conv2d (images_placeholder,kernel, [1, 1, 1, 1], padding='SAME')
#		conv = tf.nn.conv2d (images_reshaped, kernel, [1, 1, 1, 1], padding='SAME')
#		biases = tf.Variable (tf.constant(0.0, shape=[64], dtype=tf.float32),
#								trainable=True, name='biases')
#		out = tf.nn.bias_add (conv, biases)
#		conv1_1 = tf.nn.relu (out, name=scope)
#
#	pool1 = tf.nn.max_pool (conv1_1,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME',name='pool1')
#        
#
#	with tf.name_scope('conv2_1') as scope:
#		kernel = tf.Variable (tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, 
#								stddev=1e-2), name='weights')
#		conv = tf.nn.conv2d (pool1, kernel, [1, 1, 1, 1], padding='SAME')
#		biases = tf.Variable (tf.constant(0.0, shape=[128], dtype=tf.float32),
#								trainable=True, name='biases')
#		out = tf.nn.bias_add (conv, biases)
#		conv2_1 = tf.nn.relu (out, name=scope)
#
#	pool2 = tf.nn.max_pool (conv2_1,
#							ksize=[1, 2, 2, 1],
#							strides=[1, 2, 2, 1],
#							padding='SAME',
#							name='pool2')
#
#	with tf.name_scope('conv3_1') as scope:
#		kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32,
#									stddev=1e-2), name='weights')
#		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
#		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
#									trainable=True, name='biases')
#		out = tf.nn.bias_add(conv, biases)
#		conv3_1 = tf.nn.relu(out, name=scope)
#
#	pool3 = tf.nn.max_pool (conv3_1,
#							ksize=[1, 2, 2, 1],
#							strides=[1, 2, 2, 1],
#							padding='SAME',
#							name='pool3')
#
#	with tf.name_scope('conv4_1') as scope:
#		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32,
#								stddev=1e-2), name='weights')
#		conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
#		biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
#								trainable=True, name='biases')
#		out = tf.nn.bias_add(conv, biases)
#		conv4_1 = tf.nn.relu(out, name=scope)
#
#	pool4 = tf.nn.max_pool (conv4_1,
#							ksize=[1, 2, 2, 1],
#							strides=[1, 2, 2, 1],
#							padding='SAME',
#							name='pool4')	
#
#	with tf.name_scope('mentee_conv5_1') as scope:
#		kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32,
#								stddev=1e-2), name='weights')
#		conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
#		biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32),
#								 trainable=True, name='biases')
#		out = tf.nn.bias_add(conv, biases)
#		conv5_1 = tf.nn.relu(out, name=scope)
#
#	pool5 = tf.nn.max_pool (conv5_1,
#							ksize=[1, 2, 2, 1],
#							strides=[1, 2, 2, 1],
#							padding='SAME',
#							name='pool5')

       	with tf.name_scope("fc0") as scope:
		#fc0w = tf.Variable(tf.truncated_normal([IMAGE_WIDTH*IMAGE_HEIGHT, 4096],stddev=1e-2), name='weights')
        #shape = int(np.prod(pool5.get_shape()[1:]))
        	shape = int(np.prod(images_reshaped.get_shape()[1:]))
		fc0w = tf.Variable(tf.truncated_normal([shape, 102],stddev=1e-2), name='weights')
#		fc0w = tf.Variable(tf.truncated_normal([shape, 4096],stddev=1e-2), name='weights')
		fc0b = tf.Variable(tf.constant(1.0, shape=[102],dtype=tf.float32),trainable=True, name='biases')
#		pool5_flat = tf.reshape(pool5, [-1, shape])
		pool5_flat = tf.reshape(images_reshaped, [-1, shape])
		#fc0l = tf.nn.bias_add(tf.matmul(images_placeholder, fc0w), fc0b)
		fc0l = tf.nn.bias_add(tf.matmul(pool5_flat, fc0w), fc0b)
		fc0 = tf.nn.relu(fc0l)
		fc0 = tf.nn.dropout(fc0, 0.5)

	with tf.name_scope('fc1') as scope:
		#shape = int(np.prod(pool1.get_shape()[1:]))
		fc1w = tf.Variable(tf.truncated_normal([102, 102],stddev=1e-2), name='weights')
		fc1b = tf.Variable(tf.constant(1.0, shape=[102],dtype=tf.float32),trainable=True, name='biases')
		#pool1_flat = tf.reshape(pool1, [-1, shape])
		fc1l = tf.nn.bias_add(tf.matmul(fc0, fc1w), fc1b)
		fc1 = tf.nn.relu(fc1l)
		fc1 = tf.nn.dropout(fc1, 0.5)

	with tf.name_scope('fc2') as scope:
		fc2w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc2b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc2l = tf.nn.bias_add (tf.matmul (fc1, fc2w), fc2b, name = 'infer')
		fc2 = tf.nn.relu(fc2l)
		fc2 = tf.nn.dropout(fc2, 0.5)

	with tf.name_scope('fc3') as scope:
		fc3w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc3b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc3l = tf.nn.bias_add (tf.matmul (fc2, fc3w), fc3b, name = 'infer')
		fc3 = tf.nn.relu(fc3l)
	        fc3 = tf.nn.dropout(fc3, 0.5)
	
        with tf.name_scope('fc4') as scope:
		fc4w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc4b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc4l = tf.nn.bias_add (tf.matmul (fc3, fc4w), fc4b, name = 'infer')
		fc4 = tf.nn.relu(fc4l)
	        fc4 = tf.nn.dropout(fc4, 0.5)

        with tf.name_scope('fc5') as scope:
		fc5w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc5b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc5l = tf.nn.bias_add (tf.matmul (fc4, fc5w), fc5b, name = 'infer')
		fc5 = tf.nn.relu(fc5l)
	        fc5 = tf.nn.dropout(fc5, 0.5)

        with tf.name_scope('fc6') as scope:
		fc6w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc6b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc6l = tf.nn.bias_add (tf.matmul (fc5, fc6w), fc6b, name = 'infer')
		fc6 = tf.nn.relu(fc6l)
	        fc6 = tf.nn.dropout(fc6, 0.5)

        with tf.name_scope('fc7') as scope:
		fc7w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc7b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc7l = tf.nn.bias_add (tf.matmul (fc6, fc7w), fc7b, name = 'infer')
		fc7 = tf.nn.relu(fc7l)
	        fc7 = tf.nn.dropout(fc7, 0.5)

        with tf.name_scope('fc8') as scope:
		fc8w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc8b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc8l = tf.nn.bias_add (tf.matmul (fc7, fc8w), fc8b, name = 'infer')
		fc8 = tf.nn.relu(fc8l)
	        fc8 = tf.nn.dropout(fc8, 0.5)
        
        with tf.name_scope('fc9') as scope:
		fc9w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc9b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc9l = tf.nn.bias_add (tf.matmul (fc8, fc9w), fc9b, name = 'infer')
		fc9 = tf.nn.relu(fc9l)
	        fc9 = tf.nn.dropout(fc9, 0.5)
        
        with tf.name_scope('fc10') as scope:
		fc10w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc10b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc10l = tf.nn.bias_add (tf.matmul (fc9, fc10w), fc10b, name = 'infer')
		fc10 = tf.nn.relu(fc10l)
	        fc10 = tf.nn.dropout(fc10, 0.5)
        
        with tf.name_scope('fc11') as scope:
		fc11w = tf.Variable (tf.truncated_normal ([102, 102],stddev=1e-2),name='weights')
		fc11b = tf.Variable (tf.constant (1.0, shape=[102], dtype=tf.float32),trainable=True, name='biases')
                fc11l = tf.nn.bias_add (tf.matmul (fc10, fc11w), fc11b, name = 'infer')
#		fc11 = tf.nn.relu(fc11l)
#	        fc11 = tf.nn.dropout(fc11, 0.5)
	
        labels = tf.cast(labels_placeholder, tf.int32)
	oneHot = tf.one_hot (labels, 102)
#        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits (labels=labels,
#                 logits=fc3l, name="xentropy")
        loss = tf.reduce_mean (tf.square(tf.subtract(fc11l, oneHot)), name='loss')
#       print loss
    	#cost = tf.reduce_sum (tf.square (fc3l - labels), name="cost")

#	optimizer = tf.train.AdamOptimizer (LEARNING_RATE)
#	global_step = tf.Variable (0, name='global_step', trainable=False)
#	train_op = optimizer.minimize (loss, global_step=global_step, name="train")

	#optimizer = tf.train.AdamOptimizer().minimize(loss, name = "train") 
	optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss, name = "train") 
	init = tf.initialize_variables (tf.all_variables(), name='init_all_vars_op')
	tf.train.write_graph (sess.graph_def, "models/", "graph_fc_12.pb", as_text=False)
