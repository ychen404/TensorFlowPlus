# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 01:05:59 2018

@author: yitao
"""

import tensorflow as tf
import os
import sys


filename = sys.argv[1]
def printTensors(filename):

    # read pb into graph_def
    with tf.gfile.GFile(filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # import graph_def
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # print operations
    for op in graph.get_operations():
        print(op.name)

printTensors(filename)