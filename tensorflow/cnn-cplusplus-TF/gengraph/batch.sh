#!/bin/bash
# files: tom.sh

#batch_size=(128 256 512 1024)
batch_size=(128)
output=(64 128 256 512 1024 2048 4096)

#batch_size=(1 2)
#output=(3)
echo $1

#fn=sed "s/$1/new/"
#echo $fn

for i in ${batch_size[@]}
do
        for j in ${output[@]}
        do
                fn=$i"_"$j".py"
                echo $fn
		cp $1 $fn
                sed -i "s/BATCH_SIZE = 64/BATCH_SIZE = $i/" "$fn"
                sed -i "s/OUTPUT = 64/OUTPUT = $j/"  "$fn"
          
		python $fn
                rm $fn               

        done
done
