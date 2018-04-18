//
// Created by WangYingnan on 3/12/17.
//

#ifndef RSKERNELSTEST_RSMATMUL_H
#define RSKERNELSTEST_RSMATMUL_H

#include "tensorflow/contrib/android_renderscript_ops/jni/RScommon.h"
#include <iostream>
// Android log
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>


#define LOG_TAG "NDK_LOG"
#define  LOGI(...)  __android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,LOG_TAG,__VA_ARGS__)
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,LOG_TAG,__VA_ARGS__)



namespace androidrs {

namespace matmul {

static sp<RS> mRS = new RS();
static const char* cachePath = "/data/user/0/org.tensorflow.demo/cache";

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ h2;  
    }
};

//static num_flag = 0;


static std::unordered_map<std::pair<int, int>, sp<Allocation>, pair_hash> a_alloc_map;
static std::unordered_map<std::pair<int, int>, sp<Allocation>, pair_hash> b_alloc_map;
static std::unordered_map<std::pair<int, int>, sp<Allocation>, pair_hash> c_alloc_map;

static sp<ScriptIntrinsicBLAS>& initSC()
{
    static sp<ScriptIntrinsicBLAS> sc = ScriptIntrinsicBLAS::create(androidrs::matmul::mRS);
    return sc;
}

/*
  Steps for RenderScript
  1) Initialize a RenderScript Context
  2) Create at least one Allocation to be passed to a script. An Allocation is a RenderScript object that
  provides storage for fixed amount of data. Kernels in script take Allocation objects as their input
  and output.
  3) Create whatever scripts are necessary. There are two types of scripts:
       a) ScriptC
       b) ScriptIntrinsic
  4) Populate Allocations with data. 
  5) Set any necessary script globals. 
  6) Launch the appropriate kernels and invokable functions
  7) Retrieve data from Allocation objects
  8) Tear down the RenderScript context

 */
// float
static void rsMatmul_sgemm_tom(void* a_ptr, bool a_trans, void* b_ptr, bool b_trans, void* c_ptr,
                    int m, int n, int k, const float alpha, float beta)
{

    LOGI ("The alpha is %f", alpha);
    LOGI ("The beta is %f", beta);
    LOGI ("The sizes a, b and c are: %lld, %lld, %lld", sizeof(a_ptr), sizeof(b_ptr), sizeof(c_ptr));


    if(!androidrs::matmul::mRS->getContext()){
        androidrs::matmul::mRS->init(androidrs::matmul::cachePath);
    }

    if(a_alloc_map.find(std::make_pair(k, m))==a_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> a_t = Type::create(androidrs::matmul::mRS, e, k, m, 0);
        sp<Allocation> a_alloc = Allocation::createTyped(androidrs::matmul::mRS,\
                                a_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        a_alloc_map[std::make_pair(k, m)] = a_alloc;
    } else {    LOGD("a_alloc_map find");}

    if(b_alloc_map.find(std::make_pair(n, k))==b_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> b_t = Type::create(androidrs::matmul::mRS, e, n, k, 0);
        sp<Allocation> b_alloc = Allocation::createTyped(androidrs::matmul::mRS,\
                                b_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        b_alloc_map[std::make_pair(n, k)] = b_alloc;
    } else { LOGD("b_alloc_map find");}
    if(c_alloc_map.find(std::make_pair(n, m))==c_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> c_t = Type::create(androidrs::matmul::mRS, e, n, m, 0);
        sp<Allocation> c_alloc = Allocation::createTyped(androidrs::matmul::mRS,\
                                c_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        c_alloc_map[std::make_pair(n, m)] = c_alloc;
    } else {{    LOGD("c_alloc_map find");}}

    LOGD("test1");

    a_alloc_map[std::make_pair(k, m)]->copy2DRangeFrom(0, 0, k, m, a_ptr);
    b_alloc_map[std::make_pair(n, k)]->copy2DRangeFrom(0, 0, n, k, b_ptr);

    /**
     * Copy from an array into a rectangular region in this Allocation. The
     * array is assumed to be tightly packed.
     * @param[in] xoff X offset of region to update in this Allocation
     * @param[in] yoff Y offset of region to update in this Allocation
     * @param[in] w Width of region to update
     * @param[in] h Height of region to update
     * @param[in] data Array from which to copy
     */
    //void copy2DRangeFrom(uint32_t xoff, uint32_t yoff, uint32_t w, uint32_t h,
    //                     const void *data);

    LOGD("test2");
    RsBlasTranspose a_transpose = a_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;
    RsBlasTranspose b_transpose = b_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;
    LOGD("test3");

    sp<ScriptIntrinsicBLAS> script = initSC();
    LOGI ("%d, %d", a_transpose, b_transpose);

    script->SGEMM(a_transpose, b_transpose, alpha, a_alloc_map[std::make_pair(k, m)],\
                        b_alloc_map[std::make_pair(n, k)], beta, c_alloc_map[std::make_pair(n, m)]);
    LOGD("test4");
    
    if (c_ptr == NULL)
        LOGD("NULL");
    else 
        LOGD("OK");
        
    LOGD("n %d", n);
    LOGD("m %d", m);
    
    LOGD("a_ptr, %p", &a_ptr);
    LOGD("b_ptr, %p", &b_ptr);

    if (c_alloc_map.find(std::make_pair(n, m))==c_alloc_map.end()){
        LOGD("Not able to find pair");   
    } else {
        LOGD("c_alloc_map find");
        //c_alloc_map[std::make_pair(n, m)]->copy2DRangeTo(0, 0, n, m, c_ptr);
        LOGD("c_alloc_map ends");
        LOGD("c_ptr, %p", &c_ptr);
    }

    //c_alloc_map[std::make_pair(n, m)]->copy2DRangeTo(0, 0, n, m, c_ptr);
    LOGD("test5");

};  


// float
static void rsMatmul_sgemm(void* a_ptr, bool a_trans, void* b_ptr, bool b_trans, void* c_ptr,
                    int m, int n, int k, float alpha, float beta)
{

    LOGI ("rsMatmul_sgemm: The sizes a, b and c are: %lld, %lld, %lld", sizeof(a_ptr), sizeof(b_ptr), sizeof(c_ptr));

    if(!androidrs::matmul::mRS->getContext()){
        androidrs::matmul::mRS->init(androidrs::matmul::cachePath);
    }

    if(a_alloc_map.find(std::make_pair(k, m))==a_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> a_t = Type::create(androidrs::matmul::mRS, e, k, m, 0);
        sp<Allocation> a_alloc = Allocation::createTyped(androidrs::matmul::mRS, a_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        a_alloc_map[std::make_pair(k, m)] = a_alloc;
    } else { LOGD("a ptr find");}

    if(b_alloc_map.find(std::make_pair(n, k))==b_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> b_t = Type::create(androidrs::matmul::mRS, e, n, k, 0);
        sp<Allocation> b_alloc = Allocation::createTyped(androidrs::matmul::mRS, b_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        b_alloc_map[std::make_pair(n, k)] = b_alloc;
    } else { LOGD("b ptr find");}
    if(c_alloc_map.find(std::make_pair(n, m))==c_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> c_t = Type::create(androidrs::matmul::mRS, e, n, m, 0);
        sp<Allocation> c_alloc = Allocation::createTyped(androidrs::matmul::mRS, c_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        c_alloc_map[std::make_pair(n, m)] = c_alloc;
    } else {{ LOGD("c ptr find");}}

    LOGD("test1");

    a_alloc_map[std::make_pair(k, m)]->copy2DRangeFrom(0, 0, k, m, a_ptr);
    b_alloc_map[std::make_pair(n, k)]->copy2DRangeFrom(0, 0, n, k, b_ptr);

    LOGD("test2");
    RsBlasTranspose a_transpose = a_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;
    RsBlasTranspose b_transpose = b_trans ? RsBlasTranspose::RsBlasTrans : RsBlasTranspose::RsBlasNoTrans;
    LOGD("test3");

    sp<ScriptIntrinsicBLAS> script = initSC();
    LOGI ("%d, %d", a_transpose, b_transpose);

    script->SGEMM(a_transpose, b_transpose, alpha, a_alloc_map[std::make_pair(k, m)], b_alloc_map[std::make_pair(n, k)], beta, c_alloc_map[std::make_pair(n, m)]);
    LOGD("test4");
    c_alloc_map[std::make_pair(n, m)]->copy2DRangeTo(0, 0, n, m, c_ptr);
    LOGD("test5");

        /**
     * Copy from this Allocation into a rectangular region in an array. The
     * array is assumed to be tightly packed.
     * @param[in] xoff X offset of region to copy from this Allocation
     * @param[in] yoff Y offset of region to copy from this Allocation
     * @param[in] w Width of region to update
     * @param[in] h Height of region to update
     * @param[in] data destination array
     */
    //void copy2DRangeTo(uint32_t xoff, uint32_t yoff, uint32_t w, uint32_t h,
    //                   void *data);

};

}
}

#endif //RSKERNELSTEST_RSMATMUL_H
