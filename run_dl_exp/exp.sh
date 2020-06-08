#!/bin/bash

data_path=$1
gpu=$2
mode=$3
model_name=$4
#part=$6

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ]; then #|| [ -z "$5" ]; then
	echo "Plz input: data_path & gpu_idx & mode & model_name!!!!!"
	exit 0
fi

root=`pwd`
run_exp(){
	cdir=$1
	rdir=$2
	mode=$3
	cmd="cd ${cdir}"
	#cmd="${cmd}; echo ${cdir}"
	cmd="${cmd}; ./grid.sh ${gpu} ${mode} ${model_name}" 
	cmd="${cmd}; ./do-test.sh ${gpu} ${mode} ${model_name}"
	#cmd="${cmd}; ./do-pred.sh ${gpu} ${mode}"
	#cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record"
	#cmd="${cmd}; python select_params.py logs ${mode} | rev | cut -d' ' -f1-2 | rev >> ${mode}.record" # va logloss, auc
	#cmd="${cmd}; head -n10 test-score.${mode}/rnd*log >> ${mode}.record" # va logloss, auc
	#cmd="${cmd}; python cal_auc.py test-score.${mode}/  rnd_gt.svm ${pos_bias} >> ${mode}.record"
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`
for i in 'remap'
do
	cdir=${exp_dir}/${i}.${model_name}
	mkdir -p ${cdir}
	ln -sf ${root}/scripts/*.sh ${cdir}
	ln -sf ${root}/scripts/*.py ${cdir}
	ln -sf ${root}/${data_path}/${i}/*.ffm ${cdir}
	run_exp ${cdir} ${root} ${mode} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
done

