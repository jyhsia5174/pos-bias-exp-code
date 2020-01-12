#!/bin/sh

set -x
mn='dssm'

for i in 'random' 'det'
do
	cd kkbox.10.song.feat.ffm.pos.0.5.new/${i}
	echo "cd kkbox.10.song.feat.ffm.pos.0.5.new/${i}"
	#./grid.sh 0 ${mn}
	mv test_score test_score.ll
	./auto-train.sh 0 ${mn}
	./auto-pred.sh 0
	cat test_score/[0-4].log | awk '{sum+=$1} END {print "Average = ", sum/NR}'
	cd ../../
done

for i in '.comb.' '.'
do
	for j in '0.1' '0.01'
	do
		cd kkbox.10.song.feat.ffm.pos.0.5.new${i}${j}/random
		echo "cd kkbox.10.song.feat.ffm.pos.0.5.new${i}${j}/random"
		#./grid.sh 0 ${mn}
		mv test_score test_score.ll
		./auto-train.sh 0 ${mn}
		./auto-pred.sh 0
		cat test_score/[0-4].log | awk '{sum+=$1} END {print "Average = ", sum/NR}'
		cd ../../
	done
done

