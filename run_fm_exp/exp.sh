#!/bin/bash

task(){
for r in 0.5 0.2 0.1
do 
	for i in 0.1 0.01 
	do 
		for j in '.comb.' '.' 
		do 
			#cd ../data/mix/kkbox.3000${j}epsilon.${i}
			#sh ./data.sh ../../kkbox.3000.song.feat.bias $i $r
			#cd ../../../run_fm_exp
			#cd kkbox.3000${j}epsilon.${i}/rnd
			#./grid.sh 
			#./do-test.sh
			#mv logs logs.${r}
			#mv test-score test-score.${r}
			#cd ../../

			cmd="cd kkbox.3000${j}epsilon.${i}/rnd;"
			cmd="${cmd} mv logs.${r} logs;"
			cmd="${cmd} ./do-test.sh;"
			cmd="${cmd} mv logs logs.${r};"
			cmd="${cmd} mv test-score test-score.${r};"
			echo "${cmd} cd ../../"
		done; 
	done;
done
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
wait


# Run
#echo "Run"
#task | xargs -0 -d '\n' -P 2 -I {} sh -c {} 
