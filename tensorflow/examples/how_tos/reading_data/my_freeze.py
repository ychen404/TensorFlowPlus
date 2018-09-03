import tensorflow as tf
from tensorflow.python.framework import graph_util
import os,sys

output_node_names = "softmax_linear/add"
saver = tf.train.import_meta_graph('my_test_model.meta', clear_devices=True)

# tf.get_default_graph() return the default Graph being used in the current thread
graph = tf.get_default_graph()

# tf.Graph.as_default context manager, which overrides the current default graph for
# lifetime of the context
input_graph_def = graph.as_graph_def()
sess = tf.Session()
saver.restore(sess, "./my_test_model")
output_graph_def = graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            input_graph_def, # The graph_def is used to retrieve the nodes 
            output_node_names.split(",") # The output node names are used to select the usefull nodes
) 
output_graph="train-2-frozen.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())

sess.close()
