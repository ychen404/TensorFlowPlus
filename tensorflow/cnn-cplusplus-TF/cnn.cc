#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>

using namespace std;
using namespace chrono;

using tensorflow::Flag;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

string train_labels_file = "dataset_train.txt";
string test_labels_file = "dataset_test.txt";

#define VERBOSE true
#define VERBOSE_TSHD 10
#define NUM_ITERATIONS 100
#define BATCH_SIZE 25
#define IMAGE_HEIGHT 100
#define IMAGE_WIDTH 100
#define TOTAL_NUM_EXAMPLES_TRAIN 7316
#define TOTAL_NUM_EXAMPLES_TEST 1829
#define NUM_CHANNELS 3

using namespace tensorflow;

int main (int argc, char* argv[]) {

	std::string graph_definition = 	"./models/graph_simplified.pb";
	Session *session;
	GraphDef graph_def;
	SessionOptions opts;	
	std::vector<Tensor> costoutputs; // Store output
	TF_CHECK_OK (ReadBinaryProto(Env::Default(), graph_definition, &graph_def));
	
	cout << "[INFO] create a new session" << endl;
	TF_CHECK_OK (NewSession(opts, &session));

	cout << "[INFO] Load graph into session" << endl;
	TF_CHECK_OK (session->Create(graph_def));

	cout << "[INFO] Model has been loaded Successfully!" << endl;
	cout << "[INFO] Initialize our variables!" << endl;
	TF_CHECK_OK (session->Run({}, {}, {"init_all_vars_op"}, nullptr));
	cout << "[INFO] Initialization done..." << endl;

	Tensor x(DT_FLOAT, TensorShape({100, 32}));
	Tensor y(DT_FLOAT, TensorShape({100, 8}));
	auto _XTensor = x.matrix<float>();
	auto _YTensor = y.matrix<float>();

	_XTensor.setRandom();
	_YTensor.setRandom();

	for (int i = 0; i < 10; ++i) {
	TF_CHECK_OK (session->Run ({{"input", input}, {"label", label}}, {"loss"}, {}, &costOutput));
	TF_CHECK_OK (session->Run ({{"input", input}, {"label", label}}, {}, {"train"}, nullptr));
	float cost = costOutput[0].scalar<float>()(0);
	cout << "Loss is: " << cost << endl;
								costOutput.clear();

	}

	session->Close();
	delete session;
	return 0;
}

