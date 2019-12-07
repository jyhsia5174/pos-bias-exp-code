set -x
python pred.py --dataset_path ../data/random/ --model_path tmp/extdssm_lr-0.001_l2-1e-06_bs-8192.pt --model_name extdssm
python cal_revenue.py tmp.pred ../data/random/gt.svm
mv tmp.pred bidssm.pred.3

python pred.py --dataset_path ../data/random/ --model_path tmp/bidssm_lr-0.001_l2-1e-06_bs-8192.pt --model_name bidssm
python cal_revenue.py tmp.pred ../data/random/gt.svm
mv tmp.pred bidssm.pred.3

python pred.py --dataset_path ../data/random/ --model_path tmp/dssm_lr-0.001_l2-1e-06_bs-8192.pt --model_name dssm
python cal_revenue.py tmp.pred ../data/random/gt.svm
mv tmp.pred dssm.pred.3
