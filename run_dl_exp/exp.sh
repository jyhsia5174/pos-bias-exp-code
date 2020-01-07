#!/bin/sh

set -x
for i in '.comb.' '.'
do
	for j in 0.01 0.1
	do
		cd kkbox.3000${i}epsilon.${j}/rnd
		for k in 'trva' 'va' 'tr'
		do
			ln -sf ../../../data/mix/kkbox.3000${i}epsilon.${j}/select_${k}.svm ${k}.svm
		done
		ln -sf ../../../data/mix/kkbox.3000${i}epsilon.${j}/rnd_gt.svm ./
		ln -sf ../../../data/mix/kkbox.3000${i}epsilon.${j}/truth.svm ./gt.svm
		ln -sf ../../../data/mix/kkbox.3000${i}epsilon.${j}/item.svm ./
		#./grid.sh 0 ${i}
		#./auto-train.sh 0 ${i}
		#./auto-pred.sh 0
		cd ../../
	done
done
