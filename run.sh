#!/bin/sh
# A script to testing the tensorflow android app

#START = 1
#END = 3

for i in 1 2 3  
do 
# To start the activity

	echo "run $i starts"
	adb shell am start -n org.tensorflow.demo/.ClassifierActivity
	sleep 15

# To stop the activity

	adb shell am force-stop org.tensorflow.demo
	echo "run $i ends"

done
