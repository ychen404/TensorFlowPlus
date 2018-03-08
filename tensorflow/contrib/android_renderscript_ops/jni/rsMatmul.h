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
static std::unordered_map<std::pair<int, int>, sp<Allocation>, pair_hash> a_alloc_map;
static std::unordered_map<std::pair<int, int>, sp<Allocation>, pair_hash> b_alloc_map;
static std::unordered_map<std::pair<int, int>, sp<Allocation>, pair_hash> c_alloc_map;

static sp<ScriptIntrinsicBLAS>& initSC()
{
    static sp<ScriptIntrinsicBLAS> sc = ScriptIntrinsicBLAS::create(androidrs::matmul::mRS);
    return sc;
}


// float
static void rsMatmul_sgemm_tom(void* a_ptr, bool a_trans, void* b_ptr, bool b_trans, void* c_ptr,
                    int m, int n, int k, const float alpha, float beta)
{

    LOGI ("The alpha is %f", alpha);
    if(!androidrs::matmul::mRS->getContext()){
        androidrs::matmul::mRS->init(androidrs::matmul::cachePath);
    }

    if(a_alloc_map.find(std::make_pair(k, m))==a_alloc_map.end()){
        sp<const Element> e = Element::F32(androidrs::matmul::mRS);
        sp<const Type> a_t = Type::create(androidrs::matmul::mRS, e, k, m, 0);
        sp<Allocation> a_alloc = Allocation::createTyped(androidrs::matmul::mRS, a_t, RS_ALLOCATION_USAGE_SHARED | RS_ALLOCATION_USAGE_SCRIPT);
        a_alloc_map[std::make_pair(k, m)] = a_alloc;
    } else {    LOGD("a ptr find");}

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
    } else {{    LOGD("c ptr find");}}

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
};


// float
static void rsMatmul_sgemm(void* a_ptr, bool a_trans, void* b_ptr, bool b_trans, void* c_ptr,
                    int m, int n, int k, float alpha, float beta)
{

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
};

}
}

#endif //RSKERNELSTEST_RSMATMUL_H
