//
// Created by WangYingnan on 3/12/17.
//

#ifndef RSKERNELSTEST_RSCONV_H
#define RSKERNELSTEST_RSCONV_H

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include "tensorflow/contrib/android_renderscript_ops/jni/ScriptC_mScriptConv.h"


namespace androidrs {

namespace conv {

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
};

}
}


#endif //RSKERNELSTEST_RSCONV_H