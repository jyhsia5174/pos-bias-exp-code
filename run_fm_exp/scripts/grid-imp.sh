#!/bin/bash
. ./init.sh

w_train=(1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7 1e-8)
l_train=(1 4 16)
t=50
d=32
c=8

item='item.ffm'
tr='tr.ffm'
te='va.ffm'
imp='imp_tr.ffm'

#echo "item: $item; tr: $tr; te:$te; imp: $imp"
logs_path="logs"
mkdir -p $logs_path


task(){
for w in ${w_train[@]} 
do
    for l in ${l_train[@]}
    do
		cmd="./train"
		cmd="${cmd} -l $l"
		cmd="${cmd} -w $w"
		cmd="${cmd} -d $d"
		cmd="${cmd} -t ${t}"
		cmd="${cmd} -imp-r 0"
		cmd="${cmd} -c ${c}"
		cmd="${cmd} --save-model"
		cmd="${cmd} -p ${te} ${item} ${tr} ${imp} > $logs_path/${l}_${w}_${d}"
		echo "${cmd}"
    done
done
}

task
task | xargs -d '\n' -P 3 -I {} sh -c {}; rm Pva* Qva*
