#!/bin/sh

root='test_score'
model_path=`find ${root} -name "*.pt"`
model_name=`echo ${model_path} | cut -d'/' -f2 | cut -d'_' -f1`
echo "model_name: ${model_name}, model_path: ${model_path}"

set -x
python ../../main.py --dataset_path ./ --dataset_part trva --model_path ${model_path} --model_name ${model_name} --flag test --batch_size 55500 --device cuda:$1
for i in 0 1 2 3 4
do
	python ../../cal_revenue.py tmp.pred.${i} ./gt.svm | tee -a ${root}/${i}.log
	mv tmp.pred.${i} ./${root}/test.pred.${i}
done

