#!/bin/sh

set -x
mn='xdfm'
cd kkbox.10.song.feat.ffm.pos.0.5.new/${3}
./grid.sh ${2} ${mn}
./auto-train.sh ${2} ${mn}
./auto-pred.sh ${2}
cd ../../

for i in '.comb.' '.'
do
	cd kkbox.10.song.feat.ffm.pos.0.5.new${i}${1}/random
	./grid.sh ${2} ${mn}
	./auto-train.sh ${2} ${mn}
	./auto-pred.sh ${2}
	cd ../../
done

