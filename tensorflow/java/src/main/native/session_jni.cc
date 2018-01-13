/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <string.h>
#include <memory>

#include "tensorflow/c/c_api.h"
#include "tensorflow/java/src/main/native/exception_jni.h"
#include "tensorflow/java/src/main/native/session_jni.h"
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>
#include <sstream>
#include <iostream>
#include <sys/time.h>
#include <fstream>

// Android log
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>


#define LOG_TAG "JNI_LOG"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)


//#include <sys/time.h>
#include <jni.h>
//#include <stdio.h>
//#include <stdlib.h>
#include <unistd.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/errno.h>


namespace {
TF_Session* requireHandle(JNIEnv* env, jlong handle) {
  LOGD("Session_jni::requireHandle");
  static_assert(sizeof(jlong) >= sizeof(TF_Session*),
                "Cannot package C object pointers as a Java long");
  if (handle == 0) {
    throwException(env, kNullPointerException,
                   "close() has been called on the Session");
    return nullptr;
  }
  return reinterpret_cast<TF_Session*>(handle);
}

template <class T>
void resolveHandles(JNIEnv* env, const char* type, jlongArray src_array,
                    T** dst, jint n) {
//  LOGD("Session_jni::resolveHandles");
  if (env->ExceptionCheck()) return;
  jint len = env->GetArrayLength(src_array);
  if (len != n) {
    throwException(env, kIllegalArgumentException, "expected %d, got %d %s", n,
                   len, type);
    return;
  }
  jlong* src_start = env->GetLongArrayElements(src_array, nullptr);
  jlong* src = src_start;
  for (int i = 0; i < n; ++i, ++src, ++dst) {
    if (*src == 0) {
      throwException(env, kNullPointerException, "invalid %s (#%d of %d)", type,
                     i, n);
      break;
    }
    *dst = reinterpret_cast<T*>(*src);
  }
  env->ReleaseLongArrayElements(src_array, src_start, JNI_ABORT);
}

  double elapse;
  struct timespec currentTime, lastTime;

  void reportTime(const char* str) {
    std::stringstream stringstream;
    clock_gettime(CLOCK_MONOTONIC, &currentTime);
    elapse = ( currentTime.tv_sec - lastTime.tv_sec) + (double)( currentTime.tv_nsec - lastTime.tv_nsec)/1E9;
    LOGI("%s, time elapse:\t%f", str, elapse);
    //stringstream << " Time is " << elapse << " sec";
    //android_log_print(stringstream.str().c_str());
    lastTime = currentTime;
  }


  // Rewrite the timing function for reporting timestamp only with a precision of millisecond
  void currentTimeCheck(const char* str) {
    struct timeval tp;
    std::stringstream stringstream;
    gettimeofday(&tp, NULL);
    long long int ms = tp.tv_sec * 1000 + tp.tv_usec/1000;
    LOGI("%s, \t%lld", str, ms);
    //std::cout << ms << "millisecond\n";

    const char * internalStoragePath;
    char fName[64];
    
    internalStoragePath = "/mnt/sdcard";
    strcpy(fName,internalStoragePath);
    strcat(fName,"/myfile.txt");
    
    LOGI("fname = %s", fName);
    FILE *file_ptr = fopen(fName, "w+");
    fprintf (file_ptr, "The UTC time is %lld\n", ms);
    fclose(file_ptr);

    //stringstream << " Time is " << ms << " millisecond";
    //androidrs::conv::android_log_print(stringstream.str().c_str());
  }


void resolveOutputs(JNIEnv* env, const char* type, jlongArray src_op,
                    jintArray src_index, TF_Output* dst, jint n) {
  LOGD("Session_jni::resolveOutputs");
  if (env->ExceptionCheck()) return;
  jint len = env->GetArrayLength(src_op);
  if (len != n) {
    throwException(env, kIllegalArgumentException,
                   "expected %d, got %d %s Operations", n, len, type);
    return;
  }
  len = env->GetArrayLength(src_index);
  if (len != n) {
    throwException(env, kIllegalArgumentException,
                   "expected %d, got %d %s Operation output indices", n, len,
                   type);
    return;
  }
  jlong* op_handles = env->GetLongArrayElements(src_op, nullptr);
  jint* indices = env->GetIntArrayElements(src_index, nullptr);
  for (int i = 0; i < n; ++i) {
    if (op_handles[i] == 0) {
      throwException(env, kNullPointerException, "invalid %s (#%d of %d)", type,
                     i, n);
      break;
    }
    dst[i] = TF_Output{reinterpret_cast<TF_Operation*>(op_handles[i]),
                       static_cast<int>(indices[i])};
  }
  env->ReleaseIntArrayElements(src_index, indices, JNI_ABORT);
  env->ReleaseLongArrayElements(src_op, op_handles, JNI_ABORT);
}

void TF_MaybeDeleteBuffer(TF_Buffer* buf) {
  LOGD("Session_jni::TF_MaybeDeleteBuffer");

  if (buf == nullptr) return;
  TF_DeleteBuffer(buf);
}

typedef std::unique_ptr<TF_Buffer, decltype(&TF_MaybeDeleteBuffer)>
    unique_tf_buffer;

unique_tf_buffer MakeUniqueBuffer(TF_Buffer* buf) {
  return unique_tf_buffer(buf, TF_MaybeDeleteBuffer);
}

}  // namespace

JNIEXPORT jlong JNICALL Java_org_tensorflow_Session_allocate(
    JNIEnv* env, jclass clazz, jlong graph_handle) {
  LOGD("Java_org_tensorflow_Session_allocate");
  return Java_org_tensorflow_Session_allocate2(env, clazz, graph_handle,
                                               nullptr, nullptr);
}

JNIEXPORT jlong JNICALL Java_org_tensorflow_Session_allocate2(
    JNIEnv* env, jclass clazz, jlong graph_handle, jstring target,
    jbyteArray config) {
  if (graph_handle == 0) {
    throwException(env, kNullPointerException, "Graph has been close()d");
    return 0;
  }
  LOGD("Java_org_tensorflow_Session_allocate2");

  TF_Graph* graph = reinterpret_cast<TF_Graph*>(graph_handle);
  TF_Status* status = TF_NewStatus();
  TF_SessionOptions* opts = TF_NewSessionOptions();
  const char* ctarget = nullptr;
  jbyte* cconfig = nullptr;
  if (target != nullptr) {
    ctarget = env->GetStringUTFChars(target, nullptr);
  }
  if (config != nullptr) {
    cconfig = env->GetByteArrayElements(config, nullptr);
    TF_SetConfig(opts, cconfig,
                 static_cast<size_t>(env->GetArrayLength(config)), status);
    if (!throwExceptionIfNotOK(env, status)) {
      env->ReleaseByteArrayElements(config, cconfig, JNI_ABORT);
      return 0;
    }
  }
  TF_Session* session = TF_NewSession(graph, opts, status);
  if (config != nullptr) {
    env->ReleaseByteArrayElements(config, cconfig, JNI_ABORT);
  }
  if (target != nullptr) {
    env->ReleaseStringUTFChars(target, ctarget);
  }
  TF_DeleteSessionOptions(opts);
  bool ok = throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);

  return ok ? reinterpret_cast<jlong>(session) : 0;
}

JNIEXPORT void JNICALL Java_org_tensorflow_Session_delete(JNIEnv* env,
                                                          jclass clazz,
                                                          jlong handle) {
  LOGD("Java_org_tensorflow_Session_delete");

  TF_Session* session = requireHandle(env, handle);
  if (session == nullptr) return;
  TF_Status* status = TF_NewStatus();
  TF_CloseSession(session, status);
  // Result of close is ignored, delete anyway.
  TF_DeleteSession(session, status);
  throwExceptionIfNotOK(env, status);
  TF_DeleteStatus(status);
}

JNIEXPORT jbyteArray JNICALL Java_org_tensorflow_Session_run(
    JNIEnv* env, jclass clazz, jlong handle, jbyteArray jrun_options,
    jlongArray input_tensor_handles, jlongArray input_op_handles,
    jintArray input_op_indices, jlongArray output_op_handles,
    jintArray output_op_indices, jlongArray target_op_handles,
    jboolean want_run_metadata, jlongArray output_tensor_handles) {
  LOGD("Java_org_tensorflow_Session_run");
  reportTime("Java_org_tensorflow_Session_run starts");
  currentTimeCheck("Session_jni::currentTimeCheck");
  TF_Session* session = requireHandle(env, handle);
  reportTime("Session_jni::requireHandle");
  if (session == nullptr) return nullptr;

  const jint ninputs = env->GetArrayLength(input_tensor_handles);
  const jint noutputs = env->GetArrayLength(output_tensor_handles);
  const jint ntargets = env->GetArrayLength(target_op_handles);

  std::unique_ptr<TF_Output[]> inputs(new TF_Output[ninputs]);
  std::unique_ptr<TF_Tensor* []> input_values(new TF_Tensor*[ninputs]);
  std::unique_ptr<TF_Output[]> outputs(new TF_Output[noutputs]);
  std::unique_ptr<TF_Tensor* []> output_values(new TF_Tensor*[noutputs]);
  std::unique_ptr<TF_Operation* []> targets(new TF_Operation*[ntargets]);
  unique_tf_buffer run_metadata(
      MakeUniqueBuffer(want_run_metadata ? TF_NewBuffer() : nullptr));

  resolveHandles(env, "input Tensors", input_tensor_handles, input_values.get(),
                 ninputs);
  reportTime("Session_jni::resolve input tensors");
  resolveOutputs(env, "input", input_op_handles, input_op_indices, inputs.get(),
                 ninputs);
  reportTime("Session_jni::resolve input");
  resolveOutputs(env, "output", output_op_handles, output_op_indices,
                 outputs.get(), noutputs);
  reportTime("Session_jni::resolve output");
  resolveHandles(env, "target Operations", target_op_handles, targets.get(),
                 ntargets);
  reportTime("Session_jni::resolve target Operations");
  if (env->ExceptionCheck()) return nullptr;

  TF_Status* status = TF_NewStatus();

  unique_tf_buffer run_options(MakeUniqueBuffer(nullptr));
  jbyte* jrun_options_data = nullptr;
  if (jrun_options != nullptr) {
    size_t sz = env->GetArrayLength(jrun_options);
    if (sz > 0) {
      jrun_options_data = env->GetByteArrayElements(jrun_options, nullptr);
      run_options.reset(
          TF_NewBufferFromString(static_cast<void*>(jrun_options_data), sz));
    }
  }
  //LOGD("Session_jni::TFSessionRun");

  reportTime("Session_jni::TF_SessionRun starts");
  TF_SessionRun(session, run_options.get(), inputs.get(), input_values.get(),
                static_cast<int>(ninputs), outputs.get(), output_values.get(),
                static_cast<int>(noutputs), targets.get(),
                static_cast<int>(ntargets), run_metadata.get(), status);

  reportTime("Session_jni::TF_SessionRun ends");

  if (jrun_options_data != nullptr) {
    env->ReleaseByteArrayElements(jrun_options, jrun_options_data, JNI_ABORT);
  }

  if (!throwExceptionIfNotOK(env, status)) {
    TF_DeleteStatus(status);
    return nullptr;
  }
  jlong* t = env->GetLongArrayElements(output_tensor_handles, nullptr);
  for (int i = 0; i < noutputs; ++i) {
    t[i] = reinterpret_cast<jlong>(output_values[i]);
  }
  env->ReleaseLongArrayElements(output_tensor_handles, t, 0);

  jbyteArray ret = nullptr;
  if (run_metadata != nullptr) {
    ret = env->NewByteArray(run_metadata->length);
    jbyte* elems = env->GetByteArrayElements(ret, nullptr);
    memcpy(elems, run_metadata->data, run_metadata->length);
    env->ReleaseByteArrayElements(ret, elems, JNI_COMMIT);
  }
  TF_DeleteStatus(status);
  reportTime("Java_org_tensorflow_Session_run ends");
  return ret;
}
