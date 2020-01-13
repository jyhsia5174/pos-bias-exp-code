#!/bin/bash

root=`pwd`
data_path=${1}
pos_bias=0.5

if [ -z "$1" ]; then
	echo "Plz input data_path!!!!!"
	exit 0
fi

set -x
# original data
cd ${data_path}
mkdir origin
mv *.bias *.ffm *.svm origin/
cd ${root}

# generate S_c and S_t
cdir=${data_path}/derive
mkdir -p ${cdir} 
cd ${cdir}
ln -sf ${root}/scripts/trans_data_selfva.sh ./
ln -sf ${root}/scripts/chg_form.py ./
ln -sf ${root}/scripts/gen_rnd_gt.py ./
./trans_data_selfva.sh ../origin/ ${pos_bias}
cd ${root}

# generate 99%S_c + 1%S_t and 90%S_c + 10%S_t 
for i in '.comb.' '.'
do 
	for j in 0.01 0.1
	do 
		cdir=${data_path}/der${i}${j}
		mkdir -p ${cdir} 
		cd ${cdir}
		ln -sf ${root}/scripts/mix_data_selfva.sh ./
		ln -sf ${root}/scripts/gen_mix_data.py ./
		./mix_data_selfva.sh ../derive/ ${j} ${i}
		cd ${root}
	done
done

