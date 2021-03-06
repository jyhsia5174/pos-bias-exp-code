#!/bin/sh

root=$1
pos_bias=$2

set -x
for i in 'det'
do
	for j in 'ffm' 'svm'
	do
		ln -sf ${root}/${i}.${j}.pos.${pos_bias}.bias ${i}.${j}
	done
done
ln -sf ${root}/truth.* ./
ln -sf ${root}/item.* ./

num_trva=`wc -l det.ffm | cut -d' ' -f1`
num_tr=$(echo $num_trva 0.9 | awk '{ printf "%d\n" ,$1*$2}')
num_item=`wc -l item.ffm | cut -d' ' -f1`

for i in 'det'
do
	for j in 'ffm' 'svm'
	do
		python chg_form.py ${i}.${j} ${i}
		split -l ${num_tr} ${i}_trva.${j}
		mv xaa ${i}_tr.${j}
		mv xab ${i}_va.${j}
	done
done

for i in 'ffm' 'svm'
do
	for j in '.' '.const.' '.pos.'
	do
		python gen_rnd_gt.py truth.${i} rnd_gt${j}${i} ${num_item} ${j} ${pos_bias} &
	done
	wait
done
python chg_form.py truth.ffm truth
mv truth_trva.ffm gt.ffm


