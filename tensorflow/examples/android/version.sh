#!/bin/sh
# To build and install the tensorflow application

if [ "$1" = "all" ] 
then
    echo "Assembling..."
    ./gradlew assembleDebug;

    echo "Stopping the application..."
    adb shell am force-stop org.tensorflow.demo;

    echo "Installing..."
    adb install -r gradleBuild/outputs/apk/debug/android-debug.apk;

    echo "Starting..."
    adb shell am start -n org.tensorflow.demo/.ClassifierActivity

fi

if [ "$1" = "test" ]
then
    echo "Test..."
fi

if [ "$1" = "cross" ]
then
    echo "Cross-compiling..."
	cd ../../..    
	bazel build -c opt //tensorflow/contrib/android:libtensorflow_inference.so \
   --crosstool_top=//external:android/crosstool \
   --host_crosstool_top=@bazel_tools//tools/cpp:toolchain \
   --cpu=armeabi-v7a
fi

if [ "$1" = "version" ]
then 
    adb shell getprop | grep "ro.bootimage.build.fingerprint"
fi



