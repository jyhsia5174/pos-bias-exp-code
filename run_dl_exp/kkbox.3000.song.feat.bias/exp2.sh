#!/bin/sh

set -x
for i in 'bixdfm' 'extxdfm'
do
	cd det.${i}
	ln -sf ../det/*.sh ./
	ln -sf ../det/*.svm ./
	ln -sf ../det/*.lmdb ./
	./grid.sh 1 ${i}
	./auto-train.sh 1 ${i}
	./auto-pred.sh 1
	cd ../
done
