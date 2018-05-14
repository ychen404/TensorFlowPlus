#!/bin/bash
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
}
