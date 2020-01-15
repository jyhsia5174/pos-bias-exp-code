#!/bin/bash

source init.sh

w='1e-8'
l=1
t=21
d=32
c=5

item=$1
tr=$2
te=$3
imp=$4

echo "item: $item; tr: $tr; te:$te; imp: $imp"
logs_path="logs/${tr}.${te}"
mkdir -p $logs_path


task(){
echo "./train -d $d -l $l -t ${t} -w $w --imp-r -c ${c} --save-model -p ${te} ${item} ${tr} ${imp}  > $logs_path/$l.$w"
}

task
#task | xargs -d '\n' -P 10 -I {} sh -c {} &
