#!/bin/bash

data_path=$1
pos_bias=$2
gpu=$3
mode=$4
ps='wops'

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || [ -z "$4" ] ; then
	echo "Plz input: data_path & pos_bias & gpu_idx & mode!!!!!"
	exit 0
fi

root=`pwd`

run_exp(){
	cdir=$1
	rdir=$2
	mode=$3
	model_name=$4
	imp_type=$5
	cmd="cd ${cdir}"
	cmd="${cmd}; ./grid-gan.sh ${gpu} ${mode} ${model_name} ${ps} ${imp_type}" 
	cmd="${cmd}; ./do-test-gan.sh ${gpu} ${mode} ${model_name} ${ps} ${imp_type}"
	cmd="${cmd}; ./do-pred-gan.sh ${gpu} ${mode} ${ps}"
	cmd="${cmd}; echo 'va_logloss va_auc' > ${mode}.record"
	cmd="${cmd}; python select_params.py logs ${mode} | rev | cut -d' ' -f1-2 | rev >> ${mode}.record" # va logloss, auc
	cmd="${cmd}; head -n10 test-score.${mode}/rnd*log >> ${mode}.record" # va logloss, auc
	cmd="${cmd}; cd ${rdir}"
	echo ${cmd}
}

set -e
exp_dir=`basename ${data_path}`

mn='ffm'
for i in '.comb.0.01' 
do 
	for imp_type in 'gan'
	do 
		cdir=${exp_dir}/der${i}.new-${imp_type}.${mn}.${ps}
		mkdir -p ${cdir}
		ln -sf ${root}/scripts/*-gan.sh ${cdir}
		ln -sf ${root}/scripts/*.py ${cdir}
		ln -sf ${root}/${data_path}/derive/*gt*svm ${cdir}
		ln -sf ${root}/${data_path}/derive/item.svm ${cdir}
		for j in 'trva' 'tr' 
		do
			ln -sf ${root}/${data_path}/derive/select_${j}.svm ${cdir}/
			ln -sf ${root}/${data_path}/derive/det_${j}.svm ${cdir}/
			ln -sf ${root}/${data_path}/derive/random_${j}.svm ${cdir}/
		done
		ln -sf ${root}/${exp_dir}/*obs*/test-score*/*trva.pt ${cdir}/dis_trva.pt
		ln -sf ${root}/${data_path}/derive/random_va.svm ${cdir}/va.svm
		run_exp ${cdir} ${root} ${mode} ${mn} ${imp_type} | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
	done
done

## Check command
#echo "Check command list (the command may not be runned!!)"
#task
#wait


# Run
#echo "Run"
#task | xargs -0 -d '\n' -P 4 -I {} sh -c {} 
