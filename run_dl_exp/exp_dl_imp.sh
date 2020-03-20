#!/bin/bash

data_path=$1
pos_bias=$2
gpu=$3
mode=$4
model_name=$5
ps='wops'

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] || [ -z "$5" ]; then
	echo "Plz input: data_path & pos_bias & gpu_idx & mode & model_name!!!!!"
	exit 0
fi

root=`pwd`
run_exp(){
	cdir=$1
	rdir=$2
	mode=$3
	model_name=$4
	imp_dir=$5
	imp_mode=$6
	cmd="cd ${cdir}"
	cmd="${cmd}; ./grid_imp.sh ${gpu} ${mode} ${model_name} ${ps} ${imp_dir} ${imp_mode}" 
	cmd="${cmd}; ./do-test_imp.sh ${gpu} ${mode} ${model_name} ${ps} ${imp_dir} ${imp_mode}"
	cmd="${cmd}; ./do-pred_imp.sh ${gpu} ${mode} ${ps}"
	cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record"
	cmd="${cmd}; python select_params_imp.py logs ${mode} | rev | cut -d' ' -f1-2 | rev >> ${mode}.record" # va logloss, auc
	cmd="${cmd}; head -n10 test-score.${mode}/rnd*log >> ${mode}.record" # va logloss, auc
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`

for i in '.comb.'
do 
	for k in 0.1 0.01
	do 
		for m in 1 2 3 4
		do 
			cdir=${exp_dir}/der${i}${k}.imp.${model_name}.${m}
			mkdir -p ${cdir}
			ln -sf ${root}/scripts/*_imp.sh ${cdir}
			ln -sf ${root}/scripts/*_imp.py ${cdir}
			ln -sf ${root}/${data_path}/der${i}${k}.imp/*gt*svm ${cdir}
			ln -sf ${root}/${data_path}/der${i}${k}.imp/item.svm ${cdir}
			#ln -sf ${root}/${data_path}/der${i}${k}.imp/truth.svm ${cdir}
			for j in 'trva' 'tr'
			do
				cat ${root}/${data_path}/der${i}${k}.imp/select_*_${j}.svm > ${cdir}/${j}.svm
				#ln -sf ${root}/${data_path}/der${i}${k}.imp/select_sc_${j}.svm ${cdir}/${j}.svm
				ln -sf ${root}/${data_path}/der${i}${k}.imp/select_st_${j}.svm ${cdir}/imp_${j}.svm
			done
			ln -sf ${root}/${data_path}/der${i}${k}.imp/select_va.svm ${cdir}/va.svm
			run_exp ${cdir} ${root} ${mode} ${model_name} ../der.${k}.${model_name} ${m} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
		done
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
