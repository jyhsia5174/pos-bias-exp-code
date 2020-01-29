#!/bin/bash
. ./init.sh

mode=$1
./select_params.sh logs ${mode}
l=`cat ${mode}.top | awk -F ':' '{print $1}' | rev | awk -F '/' '{print $1}'| rev | awk -F '_' '{print $1}'`
w=`cat ${mode}.top | awk -F ':' '{print $1}' | rev | awk -F '/' '{print $1}'| rev | awk -F '_' '{print $2}'`
d=`cat ${mode}.top | awk -F ':' '{print $1}' | rev | awk -F '/' '{print $1}'| rev | awk -F '_' '{print $3}'`
t=`cat ${mode}.top | awk -F ':' '{print $2}' | awk -F ' ' '{print $1}'`
c=8

item='item.ffm'
tr='trva.ffm'
te_prefix='rnd_gt'
imp='imp_trva.ffm'

#echo "item: $item; tr: $tr; te:$te; imp: $imp"

logs_path="test-score.${mode}"
mkdir -p $logs_path; 

task(){
for i in '.' '.10.'
do
	for j in '' 'const.' 'pos.'
	do
		te="${te_prefix}${i}${j}ffm"
		cmd="./train"
		cmd="${cmd} -l $l"
		cmd="${cmd} -w $w"
		cmd="${cmd} -d $d"
		cmd="${cmd} -t ${t}"
		cmd="${cmd} -imp-r 0"
		cmd="${cmd} -c ${c}"
		#cmd="${cmd} --save-model"
		cmd="${cmd} -p ./${te} ./${item} ./${tr} ./${imp} > ${logs_path}/${l}_${w}_${d}${i}${j}log"
		echo "${cmd}"
	done
done
}

task
task | xargs -d '\n' -P 3 -I {} sh -c {} 
