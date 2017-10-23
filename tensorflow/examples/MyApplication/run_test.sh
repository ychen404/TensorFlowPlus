#!/bin/bash

declare -a array=(
	'"'file:///android_asset/graph_fc_1_bat_256_w_64.pb'"' 
	'"'file:///android_asset/graph_fc_1_bat_256_w_128.pb'"' 
	'"'file:///android_asset/graph_fc_1_bat_256_w_256.pb'"')


STR="file:///android_asset/graph_fc_1_bat_256_w_4096.pb"


arraylength=${#array[@]}

for i in ${array[@]}
do
	echo $i 
	#echo $STR
#	sed "s/$STR/$i/" helloWorld.txt > helloWorld_$i.txt
# If use '/' as delimiter, it throws the unkown option to 's' error 
# 'Ns' the 'N' specifies which line to replace

	sed -i "16s@.*@String str = $i;@g" /home/yitao/tensorflow_android/tensorflow/tensorflow/examples/MyApplication/app/src/main/java/com/example/yitao/myapplication/MainActivity.java
	#sed -i "s@$STR@$i@g" helloWorld.txt
# N is the line number

#	sed -i 'Ns/.*/replacement-line/' file.txt
 
	./gradlew assembleDebug

	adb install -r /home/yitao/tensorflow_android/tensorflow/tensorflow/examples/MyApplication/app/build/outputs/apk/app-debug.apk
    adb shell am start -n com.example.yitao.myapplication/.MainActivity




done



