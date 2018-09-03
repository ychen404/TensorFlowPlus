bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/home/yitao/TF_1.8/tensorflow/my_frozen_pb/graph.pb \
--input_checkpoint=/home/yitao/TF_1.8/tensorflow/my_frozen_pb/model.ckpt-0 \
--input_binary=true \
--output_graph=/home/yitao/TF_1.8/tensorflow/my_frozen_pb/graph-tf-native-tool-frozen-mobile-conv6.pb \
--output_node_names=MobilenetV1/Predictions/Reshape \