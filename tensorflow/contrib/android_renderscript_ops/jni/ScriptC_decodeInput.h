/*
 * Copyright (C) 2011-2014 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This file is auto-generated. DO NOT MODIFY!
 * The source Renderscript file: decodeInput.rs
 */

//#include "RenderScript.h"
#include "tensorflow/contrib/android_renderscript_ops/rs/cpp/RenderScript.h"

using namespace android::RSC;  // NOLINT

/* This class encapsulates access to the exported elements of the script.  Typically, you
 * would instantiate this class once, call the set_* methods for each of the exported global
 * variables you want to change, then call one of the forEach_ methods to invoke a kernel.
 */
class ScriptC_decodeInput : public android::RSC::ScriptC {
private:
    /* For each non-const variable exported by the script, we have an equivalent field.  This
     * field contains the last value this variable was set to using the set_ method.  This may
     * not be current value of the variable in the script, as the script is free to modify its
     * internal variable without changing this field.  If the script initializes the
     * exported variable, the constructor will initialize this field to the same value.
     */
    int32_t mExportVar_inputRows;
    int32_t mExportVar_inputCols;
    int32_t mExportVar_padRows;
    int32_t mExportVar_padCols;
    int32_t mExportVar_decodeStride;
    int32_t mExportVar_startIdx;
    android::RSC::sp<android::RSC::Allocation> mExportVar_allPtrF32;
    /* The following elements are used to verify the types of allocations passed to kernels.
     */
    android::RSC::sp<const android::RSC::Element> __rs_elem_F32;
public:
    ScriptC_decodeInput(android::RSC::sp<android::RSC::RS> rs);
    virtual ~ScriptC_decodeInput();

    /* Methods to set and get the variables exported by the script. Const variables will not
     * have a setter.
     * 
     * Note that the value returned by the getter may not be the current value of the variable
     * in the script.  The getter will return the initial value of the variable (as defined in
     * the script) or the the last value set by using the setter method.  The script is free to
     * modify its value independently.
     */
    void set_inputRows(int32_t v) {
        setVar(0, &v, sizeof(v));
        mExportVar_inputRows = v;
    }

    int32_t get_inputRows() const {
        return mExportVar_inputRows;
    }

    void set_inputCols(int32_t v) {
        setVar(1, &v, sizeof(v));
        mExportVar_inputCols = v;
    }

    int32_t get_inputCols() const {
        return mExportVar_inputCols;
    }

    void set_padRows(int32_t v) {
        setVar(2, &v, sizeof(v));
        mExportVar_padRows = v;
    }

    int32_t get_padRows() const {
        return mExportVar_padRows;
    }

    void set_padCols(int32_t v) {
        setVar(3, &v, sizeof(v));
        mExportVar_padCols = v;
    }

    int32_t get_padCols() const {
        return mExportVar_padCols;
    }

    void set_decodeStride(int32_t v) {
        setVar(4, &v, sizeof(v));
        mExportVar_decodeStride = v;
    }

    int32_t get_decodeStride() const {
        return mExportVar_decodeStride;
    }

    void set_startIdx(int32_t v) {
        setVar(5, &v, sizeof(v));
        mExportVar_startIdx = v;
    }

    int32_t get_startIdx() const {
        return mExportVar_startIdx;
    }

    void bind_allPtrF32(android::RSC::sp<android::RSC::Allocation> v) {
        bindAllocation(v, 6);
        mExportVar_allPtrF32 = v;
    }

    android::RSC::sp<android::RSC::Allocation> get_allPtrF32() const {
        return mExportVar_allPtrF32;
    }

    // No forEach_root(...)
    /* For each kernel of the script corresponds one method.  That method queues the kernel
     * for execution.  The kernel may not have completed nor even started by the time this
     * function returns.  Calls that extract the data out of the output allocation will wait
     * for the kernels to complete.
     */
    void forEach_decode_F32(android::RSC::sp<android::RSC::Allocation> aout);
};

