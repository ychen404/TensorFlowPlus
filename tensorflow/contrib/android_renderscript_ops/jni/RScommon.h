#ifndef RSKERNELSTEST_RSCOMMON_H
#define RSKERNELSTEST_RSCOMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <unordered_map>
#include <android/log.h>
#include "tensorflow/contrib/android_renderscript_ops/rs/cpp/RenderScript.h"

using namespace android::RSC;

// static bool Android_GetPackageName(char* outPackageName, size_t length) {
//     Android_App* app = Android_GetApp();
//     ANativeActivity*activity = app->activity;

//     JNIEnv* env = activity->env;
//     //note: we need to attach dalvik VM to current thread, as it is not main thread
//     JavaVM* vm = activity->vm;
//     if ( (*vm)->GetEnv(vm, (void **)&env, JNI_VERSION_1_6) < 0 )
//         (*vm)->AttachCurrentThread(vm, &env, NULL);

//     //get package name from Activity Class(context)
//     jclass android_content_Context = (*env)->GetObjectClass(env, activity->clazz);
//     jmethodID midGetPackageName = (*env)->GetMethodID(env, android_content_Context, "getPackageName", "()Ljava/lang/String;");
//     jstring PackageName= (jstring)(*env)->CallObjectMethod(env, activity->clazz, midGetPackageName);

//     bool ret = false;
//     if( PackageName != null ) {
//         // get UTF8 string & copy to dest
//         const char* charBuff = (*env)->GetStringUTFChars(env, PackageName, NULL);
//         strncpy(outPackageName, charBuff, length);
//         outPackageName[length-1]='\0';

//         (*env)->ReleaseStringUTFChars(PackageName, charBuff);
//         (*env)->DeleteLocalRef(env, PackageName);
//     }
//     (*env)->DeleteLocalRef(env, android_content_Context);

//     return ret;
// }

#endif //RSKERNELSTEST_RSCOMMON_H