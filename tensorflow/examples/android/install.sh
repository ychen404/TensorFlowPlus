#!/bin/sh
# To build and install the tensorflow application

echo "Assembling..."
./gradlew assembleDebug;

echo "Stopping the application..."
adb shell am force-stop org.tensorflow.demo;

echo "Installing..."
adb install -r gradleBuild/outputs/apk/debug/android-debug.apk;

echo "Starting..."
adb shell am start -n org.tensorflow.demo/.ClassifierActivity
