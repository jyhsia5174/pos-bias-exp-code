#!/bin/bash

root=$1
rnd_ratio=$2
mode=$3

set -x
python gen_mix_data.py $1 $2 $3

#for i in 'svm' 'ffm'
#do
#	ln -sf select_trva.${i} select_tr.${i}
#	ln -sf ${root}/random_va.${i} select_va.${i}
#done

ln -sf ${root}/*gt.*m ./
ln -sf ${root}/item.* ./
ln -sf ${root}/truth.svm ./

num_trva=`wc -l select_trva.ffm | cut -d' ' -f1`
num_tr=$(echo $num_trva 0.9 | awk '{ printf "%d\n" ,$1*$2}')

for j in 'ffm' 'svm'
do
	split -l ${num_tr} select_trva.${j}
	mv xaa select_tr.${j}
	mv xab select_va.${j}
done
