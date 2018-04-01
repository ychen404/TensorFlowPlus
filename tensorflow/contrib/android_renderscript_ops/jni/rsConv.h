//
// Created by WangYingnan on 3/12/17.
//

#ifndef RSKERNELSTEST_RSCONV_H
#define RSKERNELSTEST_RSCONV_H
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define LOG_TAG "NDK_LOG"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/ScriptC_mScriptConv.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/ScriptC_decodeFilter.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/ScriptC_decodeInput.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/ScriptC_decodeOutput.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/ScriptC_utils.h"

namespace androidrs {

namespace conv {

void reportTime(const char* str);
struct rsConvInfo{
    int in_depth;
    int input_rows;
    int input_cols;

    int filter_rows;
    int filter_cols;

    int stride_rows;
    int stride_cols;

    int pad_rows;
    int pad_cols;

    int out_depth;
    int out_rows;
    int out_cols;

    int batch;
    int data_format; // 4 F32, 1 U8

    int filter_sz;
    int input_sz;
    int output_sz;

    double elapse;
    struct timespec currentTime, lastTime;

    void reportTime(const char* str) {
        clock_gettime(CLOCK_MONOTONIC, &currentTime);
        elapse = ( currentTime.tv_sec - lastTime.tv_sec) + (double)( currentTime.tv_nsec - lastTime.tv_nsec)/1E9;
        LOGI("%s, time elapse:\t%f", str, elapse);
        lastTime = currentTime;

    }

    rsConvInfo(int n1, int n2, int n3, int n4, int n5, int n6, int n7, int n8, int n9, int n10, int n11, int n12, int n13, int n14){
        in_depth=n1;
        input_rows=n2;
        input_cols=n3;
        filter_rows=n4;filter_cols=n5;
        stride_rows=n6;stride_cols=n7;
        pad_rows=n8;pad_cols=n9;
        out_depth=n10;out_rows=n11;out_cols=n12;
        batch=n13;data_format=n14;

        filter_sz = out_depth * in_depth * filter_rows * filter_cols;
        input_sz = in_depth * input_rows * input_cols;
        output_sz = out_depth * out_rows * out_cols;
    };
};

    static sp<RS> mRS = new RS();
    static const char* cachePath = "/data/user/0/org.tensorflow.demo/cache";
    static std::unordered_map<int, sp<Allocation>> allFilters_alloc_map;
    static std::unordered_map<int, sp<Allocation>> allInputs_alloc_map;
    static std::unordered_map<int, sp<Allocation>> allOutputs_alloc_map;

sp<ScriptC_mScriptConv>& initSC()
{
    static sp<ScriptC_mScriptConv> sc = new ScriptC_mScriptConv(androidrs::conv::mRS);
    return sc;
}

template <typename T>
void rsConv_script(void* filter, void* input, void* output, rsConvInfo convInfo)
{

    //reportTime("rsConv_script::start");
    if(!androidrs::conv::mRS->getContext()){
        androidrs::conv::mRS->init(androidrs::conv::cachePath);
    }

    if(allFilters_alloc_map.find(convInfo.filter_sz)==allFilters_alloc_map.end()){
        static sp<const Element> e = Element::F32(androidrs::conv::mRS);
        sp<const Type> all_filters_t = Type::create(androidrs::conv::mRS, e, convInfo.filter_sz, 0, 0);
        sp<Allocation > allFilters_alloc = Allocation::createTyped(androidrs::conv::mRS, all_filters_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        allFilters_alloc_map[convInfo.filter_sz] = allFilters_alloc;
    }
    if(allInputs_alloc_map.find(convInfo.input_sz)==allInputs_alloc_map.end()){
        static sp<const Element> e = Element::F32(androidrs::conv::mRS);
        sp<const Type> all_inputs_t = Type::create(androidrs::conv::mRS, e, convInfo.input_sz, 0, 0);
        sp<Allocation > allInputs_alloc = Allocation::createTyped(androidrs::conv::mRS, all_inputs_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        allInputs_alloc_map[convInfo.input_sz] = allInputs_alloc;
    }
    if(allOutputs_alloc_map.find(convInfo.output_sz)==allOutputs_alloc_map.end()){
        static sp<const Element> e = Element::F32(androidrs::conv::mRS);
        sp<const Type> all_outputs_t = Type::create(androidrs::conv::mRS, e, convInfo.output_sz, 0, 0);
        sp<Allocation > allOutputs_alloc = Allocation::createTyped(androidrs::conv::mRS, all_outputs_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        allOutputs_alloc_map[convInfo.output_sz] = allOutputs_alloc;
    }

    allFilters_alloc_map[convInfo.filter_sz]->copy1DFrom(filter);
    allInputs_alloc_map[convInfo.input_sz]->copy1DFrom(input);

    sp<ScriptC_mScriptConv> sc = initSC();

    //kernel
    sc->set_in_depth(convInfo.in_depth);
    sc->set_input_rows(convInfo.input_rows);
    sc->set_input_cols(convInfo.input_cols);
    sc->set_filter_rows(convInfo.filter_rows);
    sc->set_filter_cols(convInfo.filter_cols);
    sc->set_stride_rows(convInfo.stride_rows);
    sc->set_stride_cols(convInfo.stride_cols);
    sc->set_pad_rows(convInfo.pad_rows);
    sc->set_pad_cols(convInfo.pad_cols);
    sc->set_out_depth(convInfo.out_depth);
    sc->set_out_rows(convInfo.out_rows);
    sc->set_out_cols(convInfo.out_cols);

    sc->set_filters(allFilters_alloc_map[convInfo.filter_sz]);
    sc->set_inputs(allInputs_alloc_map[convInfo.input_sz]);
    sc->invoke_initParam();

    sc->forEach_launchConvF32(allOutputs_alloc_map[convInfo.output_sz]);

    // sync
    allOutputs_alloc_map[convInfo.output_sz]->copy1DTo(output);
    //reportTime("rsConv_script::ends");

};




template <typename T>
void rsConv_intrinsic(void* filter, void* input, void* output, rsConvInfo convInfo)
{

   reportTime("rsConv_intrinsic::start");
      // assume square filter
      const size_t filter_w = convInfo.filter_rows;
      const size_t filter_sz = filter_w * filter_w;
      const size_t filter_stride_e = convInfo.out_depth * convInfo.in_depth;
      const size_t input_stride_e = convInfo.in_depth;
      const size_t padded_rows = convInfo.input_rows + 2*convInfo.pad_rows;
      const size_t padded_cols = convInfo.input_cols + 2*convInfo.pad_cols;
  
      if(!androidrs::conv::mRS->getContext()){
          androidrs::conv::mRS->init(androidrs::conv::cachePath);
      }
      reportTime("mRs->getContext");
      sp<const Element> e, ef;
  
      ef = Element::F32(androidrs::conv::mRS);
      if(convInfo.data_format==4){
          e = Element::F32(androidrs::conv::mRS);
      }else if(convInfo.data_format==1){
          e = Element::U8(androidrs::conv::mRS);
      }
      reportTime("element");
      size_t e_bytes = e->getSizeBytes();
      reportTime("getSizeBytes");
      // decode filters
      sp<const Type> all_filters_t = Type::create(androidrs::conv::mRS, e, filter_stride_e*filter_sz,
                                                         0,
                                                         0);
      reportTime("Type::create decodeFilters");
  
      sp<Allocation > allFilters_alloc = Allocation::createTyped(androidrs::conv::mRS, all_filters_t,
                                                                 RS_ALLOCATION_USAGE_SHARED |
                                                                 RS_ALLOCATION_USAGE_SCRIPT);
      reportTime("allFilters_alloc");
      allFilters_alloc->copy1DFrom(filter);
      reportTime("copy1DFrom");
  
      sp<const Type> one_filter_t = Type::create(androidrs::conv::mRS, ef, filter_sz,
                                                        0,
                                                        0);
      reportTime("one_filter_t");
      // std::vector<std::vector<float* > > mFilters2D(
      //     convInfo.out_depth, std::vector<float* >(convInfo.in_depth, nullptr)
      // );
      // TODO: U8 mode 
      std::vector<sp<Allocation>> mFilters2D;
      reportTime("mFilter2D");
      for(size_t i=0;i<convInfo.in_depth;++i){
          for(size_t j=0;j<convInfo.out_depth;++j){
              sp<Allocation> one_filter = Allocation::createTyped(androidrs::conv::mRS, one_filter_t,
                                                                  RS_ALLOCATION_USAGE_SHARED
                                                                  | RS_ALLOCATION_USAGE_SCRIPT);
              mFilters2D.push_back(one_filter);
  
              sp<ScriptC_decodeFilter> sc = new ScriptC_decodeFilter(androidrs::conv::mRS);
              sc->set_filterW(filter_w);
              sc->set_decodeStride(filter_stride_e);
              sc->set_startIdx(i * convInfo.out_depth + j);
              sc->bind_allPtrF32(allFilters_alloc);
              sc->forEach_decode_F32(one_filter);
  
              // auto filter_transposed = static_cast<T*>(filter) + (i * convInfo.out_depth + j);
              // mFilters2D[j][i] = new float[filter_sz];
              // auto filter_transposed = static_cast<T*>(filter) + (i * convInfo.out_depth + j);
              // mFilters2D[j][i] = new float[filter_sz];
              // for(size_t p=0;p<filter_w;++p){
              //     for(size_t q=0;q<filter_w;++q){
              //         mFilters2D[j][i][p * filter_w + q] = \
              //          (float)filter_transposed[(q * filter_w + p) * filter_stride_e];
              //     }
              // }
          }
      }
      reportTime("mFilter2D for loop");
      androidrs::conv::mRS->finish();
      reportTime("mRS->finish");
      // for(int i=0;i<convInfo.out_depth;++i){
      //     for(int j = 0;j<convInfo.in_depth;++j){
      //         for(int k=0;k<9;++k){
      //             LOGI("%f", static_cast<float*>(mFilters2D[j*convInfo.out_depth+i]->getPointer())[k]);
      //         }
      //         LOGE("one sub filter"); 
      //     }
      //     LOGE("one filter");
      // }
      // return;
  
      // decode input
      // auto input_cast = static_cast<T*>(input);
      sp<const Type> all_inputs_t = Type::create(androidrs::conv::mRS, e, \
       convInfo.in_depth*convInfo.input_rows*convInfo.input_cols,
                                                         0,
                                                         0);
      reportTime("Type::create decodeInputs");
  
      sp<Allocation > allInputs_alloc = Allocation::createTyped(androidrs::conv::mRS, all_inputs_t,
                                                                RS_ALLOCATION_USAGE_SHARED |
                                                                RS_ALLOCATION_USAGE_SCRIPT);
      reportTime("Type::allocate decode inputs");
      allInputs_alloc->copy1DFrom(input);
      reportTime("decodeInputs::copy1DFrom");
      sp<const Type> input_layer_t = Type::create(androidrs::conv::mRS, e, padded_cols,
                                                         padded_rows,
                                                         0);
      reportTime("create:: input layers");
      std::vector<sp<Allocation > > intput_layers;
      reportTime("vector input_layers");
      for(size_t k=0;k<convInfo.in_depth;++k){
          sp<Allocation > input_alloc = Allocation::createTyped(androidrs::conv::mRS, input_layer_t,
                                                                RS_ALLOCATION_USAGE_SHARED |
                                                                RS_ALLOCATION_USAGE_SCRIPT);
          intput_layers.push_back(input_alloc);
  
          sp<ScriptC_decodeInput> sc = new ScriptC_decodeInput(androidrs::conv::mRS);
          sc->set_inputRows(convInfo.input_rows);
          sc->set_inputCols(convInfo.input_cols);
          sc->set_padRows(convInfo.pad_rows);
          sc->set_padCols(convInfo.pad_cols);
          sc->set_decodeStride(input_stride_e);
          sc->set_startIdx(k);
          sc->bind_allPtrF32(allInputs_alloc);
          sc->forEach_decode_F32(input_alloc);
  
      }
      reportTime("input_alloc");
      androidrs::conv::mRS->finish();
      reportTime("decodeInput::finish");
      // Conv
      sp<const Type> output_layer_t = Type::create(androidrs::conv::mRS, e, padded_cols,
                                                          padded_rows,
                                                          0);
      reportTime("create output layer");
      std::vector<std::vector<sp<Allocation> > > output_filters_reponse(
          convInfo.out_depth, std::vector<sp<Allocation> >(convInfo.in_depth, NULL)
      );
      reportTime("output_filters_response");
      if(filter_w==3){
          for(size_t i=0;i<convInfo.out_depth;++i){       
              for(size_t j=0;j<convInfo.in_depth;++j){
                  sp<Allocation > output_alloc_filter = \
                  Allocation::createTyped(androidrs::conv::mRS, output_layer_t,
                                          RS_ALLOCATION_USAGE_SHARED |
                                          RS_ALLOCATION_USAGE_SCRIPT);
                  output_filters_reponse[i][j] = output_alloc_filter;
  
                  sp<ScriptIntrinsicConvolve3x3>
                          sc = ScriptIntrinsicConvolve3x3::create(androidrs::conv::mRS, e);
                  sc->setCoefficients(
                      static_cast<float*>(mFilters2D[j*convInfo.out_depth+i]->getPointer())
                  );
                  sc->setInput(intput_layers[j]);
                  sc->forEach(output_alloc_filter);
              }
          }
          reportTime("filter_w == 3");
      }else if(filter_w==5){
          for(size_t i=0;i<convInfo.out_depth;++i){       
              for(size_t j=0;j<convInfo.in_depth;++j){
                  sp<Allocation >
                          output_alloc_filter = Allocation::createTyped(androidrs::conv::mRS,
                                                                        output_layer_t,
                                                                        RS_ALLOCATION_USAGE_SHARED |
                                                                        RS_ALLOCATION_USAGE_SCRIPT);
                  output_filters_reponse[i][j] = output_alloc_filter;
  
                  sp<ScriptIntrinsicConvolve5x5>
                          sc = ScriptIntrinsicConvolve5x5::create(androidrs::conv::mRS, e);
                  sc->setCoefficients(
                      static_cast<float*>(mFilters2D[j*convInfo.out_depth+i]->getPointer())
                  );
                  sc->setInput(intput_layers[j]);
                  sc->forEach(output_alloc_filter);
              }
          }
      }
      androidrs::conv::mRS->finish();
      reportTime("output layer finish");
      // sum up
      std::vector<sp<Allocation>> output_alloc_final;
      reportTime("output_alloc_final");
      sp<ScriptC_utils> sc = new ScriptC_utils(androidrs::conv::mRS);
      reportTime("ScriptC_utils");
      if(convInfo.data_format==4){
          for(size_t i=0;i<output_filters_reponse.size();++i){
              sp<Allocation >
                      output_alloc_filter = Allocation::createTyped(androidrs::conv::mRS, \
                      output_layer_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
              output_alloc_final.push_back(output_alloc_filter);
              for(size_t j=0;j<output_filters_reponse[i].size();++j){
                  sc->forEach_sumAlloc_F32(output_filters_reponse[i][j], output_alloc_filter);
                  androidrs::conv::mRS->finish();
              }
          }
      }
      else if(convInfo.data_format==1){
          for(size_t i=0;i<output_filters_reponse.size();++i){
              sp<Allocation >
                      output_alloc_filter = Allocation::createTyped(androidrs::conv::mRS, output_layer_t,
                                                                    RS_ALLOCATION_USAGE_SHARED |
                                                                    RS_ALLOCATION_USAGE_SCRIPT);
              output_alloc_final.push_back(output_alloc_filter);
              for(size_t j=0;j<output_filters_reponse[i].size();++j){
                  sc->forEach_sumAlloc_U8(output_filters_reponse[i][j], output_alloc_filter);
                  androidrs::conv::mRS->finish();
              }
          }
      }
      reportTime("data_format conditions");
      //output
      sp<const Type> all_outputs_t = Type::create(androidrs::conv::mRS, e, \
      convInfo.out_depth*convInfo.out_rows*convInfo.out_cols,
                                                         0,
                                                         0);
      reportTime("create::all_outputs_t::element");
      sp<Allocation >
              allOutputs_alloc = Allocation::createTyped(androidrs::conv::mRS, all_outputs_t, \
              RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
      reportTime("allOutputs_alloc");
      for(int k=0;k<output_alloc_final.size();++k){
  
          sp<ScriptC_decodeOutput> sc = new ScriptC_decodeOutput(androidrs::conv::mRS);
          sc->set_inputRows(convInfo.input_rows);
          sc->set_inputCols(convInfo.input_cols);
          sc->set_padRows(convInfo.pad_rows);
          sc->set_padCols(convInfo.pad_cols);
          sc->set_outDepth(convInfo.out_depth);
          sc->set_strideRows(convInfo.stride_rows);
          sc->set_strideCols(convInfo.stride_cols);
          sc->set_outCols(convInfo.out_cols);
          sc->set_startIdx(k);
  
          sc->set_layerInPtrF32(output_alloc_final[k]);
          sc->bind_allOutPtrF32(allOutputs_alloc);
          sc->forEach_decode_F32(output_alloc_final[k]);
  
      }
      reportTime("decodeOutput");
      allOutputs_alloc->copy1DTo(output);
      reportTime("allOutputs_alloc->copy1DTo_output");

};


}
}


#endif //RSKERNELSTEST_RSCONV_H