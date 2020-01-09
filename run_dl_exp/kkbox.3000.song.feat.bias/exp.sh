#!/bin/sh

set -x
for i in 'biffm' 'extffm'
do
	cd det.${i}
	ln -sf ../det/*.sh ./
	ln -sf ../det/*.svm ./
	ln -sf ../det/*.lmdb ./
	./grid.sh 0 ${i}
	./auto-train.sh 0 ${i}
	./auto-pred.sh 0
	cd ../
done
