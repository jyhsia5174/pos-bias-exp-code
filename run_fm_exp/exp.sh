#!/bin/bash

data_path=$1
mode=$2

if [ -z "$1" ] && [ -z "$2" ]; then
	echo "Plz input data_path & mode!!!!!"
	exit 0
fi

root=`pwd`
pos_bias=0.5

run_exp(){
	cdir=$1
	rdir=$2
	mode=$3
	cmd="cd ${cdir}"
	#cmd="${cmd}; echo ${cdir}"
	cmd="${cmd}; ./grid.sh" 
	cmd="${cmd}; ./do-test.sh ${mode}"
	#cmd="${cmd}; ./auto-pred.sh"
	cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record"
	cmd="${cmd}; python select_params.py logs ${mode} | rev | cut -d' ' -f1-2 | rev >> ${mode}.record"
	cmd="${cmd}; python cal_auc.py test-score.${mode}/  rnd_gt.ffm ${pos_bias} >> ${mode}.record"
	#cmd="${cmd}; tail -n2 test-score.${mode}/*.log | rev | cut -d' ' -f1 | rev"
	#cmd="${cmd}; cat test-score/[0-4].log | awk '{sum+=\$1} END {print \"Average = \", sum/NR}'"
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`
for i in 'det' 'random'
do
	cdir=${exp_dir}/derive.${i}
	mkdir -p ${cdir}
	ln -sf ${root}/scripts/*.sh ${cdir}
	ln -sf ${root}/scripts/*.py ${cdir}
	ln -sf ${root}/hybrid-ocffm/train ${cdir}/hytrain
	ln -sf ${root}/${data_path}/derive/*gt*ffm ${cdir}
	ln -sf ${root}/${data_path}/derive/item.ffm ${cdir}
	for j in 'trva' 'tr' 'va'
	do
		ln -sf ${root}/${data_path}/derive/${i}_${j}.ffm ${cdir}/${j}.ffm
	done
	run_exp ${cdir} ${root} ${mode} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
done

for i in '.comb.' '.'
do 
	for k in 0.01 0.1
	do 
		cdir=${exp_dir}/der${i}${k}
		mkdir -p ${cdir}
		ln -sf ${root}/scripts/*.sh ${cdir}
		ln -sf ${root}/scripts/*.py ${cdir}
		ln -sf ${root}/hybrid-ocffm/train ${cdir}/hytrain
		ln -sf ${root}/${data_path}/der${i}${k}/*gt*ffm ${cdir}
		ln -sf ${root}/${data_path}/der${i}${k}/item.ffm ${cdir}
		for j in 'trva' 'tr' 'va'
		do
			ln -sf ${root}/${data_path}/der${i}${k}/select_${j}.ffm ${cdir}/${j}.ffm
		done
		run_exp ${cdir} ${root} ${mode} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	done
done
exit 0


## Check command
#echo "Check command list (the command may not be runned!!)"
#task
#wait


# Run
#echo "Run"
#task | xargs -0 -d '\n' -P 4 -I {} sh -c {} 
