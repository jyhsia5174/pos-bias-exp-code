set -x

#python pred.py --dataset_path ../data/random-top100/ --dataset_part trva --model_path top100/dssm_lr-0.001_l2-1e-06_bs-8192_trva-nopos.pt --model_name dssm
python ../../pred.py ./
for i in 0 1 2 3 4
do
	python ../../cal_revenue.py tmp.pred.${i} gt.ffm 0.5 | tee test-score/${i}.log
	mv tmp.pred.${i} test-score/test.pred.${i}
done 
