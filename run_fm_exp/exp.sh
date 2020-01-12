#!/bin/bash

mode=${1}
#set -x
task(){
for r in 0.1 #0.2 0.5
do 
	for j in '.comb.' '.' 
	do 
		for i in 0.01 0.1 
		do 
			#cd ../data/mix/kkbox.3000${j}epsilon.${i}
			#sh ./data.sh ../../kkbox.3000.song.feat.bias $i $r
			#cd ../../../run_fm_exp
			cd kkbox.10.song.feat.ffm.pos.0.5.new${j}${i}/random
			echo ${j}${i}
			#./grid.sh 
			#rm -r test-score*
			mv test-score test-score.${r}.ll
			./do-test.sh ${mode}
			./auto-pred.sh
			python ../../select_params.py logs ${mode} 
			tail -n2 test-score/*.*.log
			cat test-score/[0-4].log | awk '{sum+=$1} END {print "Average = ", sum/NR}'
			#mv logs logs.${r}
			cd ../../

			#cmd="cd kkbox.3000${j}epsilon.${i}/rnd;"
			#cmd="${cmd} mv logs.${r} logs;"
			#cmd="${cmd} mv test-score.${r} test-score.${r}.auc;"
			#cmd="${cmd} ./do-test.sh;"
			#cmd="${cmd} mv logs logs.${r};"
			#cmd="${cmd} mv test-score test-score.${r}.ll;"
			#echo "${cmd} cd ../../"

			#cmd="cd kkbox.3000${j}epsilon.${i}/rnd;"
			#cmd="${cmd} mv logs.${r} logs;"
			#cmd="${cmd} ./do-test.sh;"
			#cmd="${cmd} mv logs logs.${r};"
			#cmd="${cmd} ./auto-pred.sh;"
			#cmd="${cmd} mv Pva* Qva* test-score/;"
			#echo "${cmd} cd ../../"
		done; 
	done;
	for j in 'det' 'random'
	do
		cd kkbox.10.song.feat.ffm.pos.0.5.new/${j}
		echo ${j}
		#./grid.sh 
		mv test-score test-score.ll
		#rm -r test-score*
		./do-test.sh ${mode} 
		./auto-pred.sh
		python ../../select_params.py logs ${mode} 
		tail -n2 test-score/*.*.log
		cat test-score/[0-4].log | awk '{sum+=$1} END {print "Average = ", sum/NR}'
		cd ../../
	done
done
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
wait


# Run
#echo "Run"
#task | xargs -0 -d '\n' -P 4 -I {} sh -c {} 
