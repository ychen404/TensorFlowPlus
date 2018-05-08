#!/bin/bash

declare -a array=(
	'"'file:///android_asset/graph_fc_2_bat_256_w_64.pb'"' 
	'"'file:///android_asset/graph_fc_2_bat_256_w_128.pb'"' 
	'"'file:///android_asset/graph_fc_2_bat_256_w_256.pb'"'
	'"'file:///android_asset/graph_fc_2_bat_256_w_512.pb'"'
	'"'file:///android_asset/graph_fc_2_bat_256_w_1024.pb'"'
	'"'file:///android_asset/graph_fc_2_bat_256_w_2048.pb'"'
	'"'file:///android_asset/graph_fc_2_bat_256_w_4096.pb'"'
	)


STR="file:///android_asset/graph_fc_1_bat_256_w_4096.pb"

arraylength=${#array[@]}

for i in ${array[@]}
do
	echo $i 
	#echo $STR
#	sed "s/$STR/$i/" helloWorld.txt > helloWorld_$i.txt
# If use '/' as delimiter, it throws the unkown option to 's' error 
# 'Ns' the 'N' specifies which line to replace

	sed -i "61s@.*@private static final String MODEL_FILE = $i;@g" /home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/src/org/tensorflow/demo/ClassifierActivity.java
	#sed -i "s@$STR@$i@g" helloWorld.txt
# N is the line number

#	sed -i 'Ns/.*/replacement-line/' file.txt
 
	./gradlew assembleDebug
	
	adb install -r /home/yitao/tensorflow_android/tensorflow/tensorflow/examples/android/gradleBuild/outputs/apk/android-debug.apk
    
    for j in 1 2 3 
    {
    	echo "run $i $j starts" 
    	adb shell am start -n org.tensorflow.demo/.ClassifierActivity
	    sleep 15
        PID=$(adb shell ps | grep "org.tensorflow.demo" | awk '{print $6}')
        echo $PID
        while [ "$PID" != "futex_wait" ]
        do
        	sleep 5
        	echo "still running"
 		    PID=$(adb shell ps | grep "org.tensorflow.demo" | awk '{print $6}')
 		    echo "$PID"
    		if [ "$PID" == "futex_wait" ]
    		then 
    			sleep 5
    			echo "ready to kill"
    			break
    		else
    			continue
    		fi 
    	done
    	
    	adb shell am force-stop org.tensorflow.demo
    	echo "run $i $j ends"

	 	
    }


done



