#!/bin/bash

root=$1
rnd_ratio=$2

set -x
python sc_st_split.py $1 $2

for i in 'ffm' 'svm'
do
	ln -sf ${root}/random_va.ffm select_va.${i}
	ln -sf ${root}/*gt*${i} ./
	ln -sf ${root}/item.${i} ./
done
