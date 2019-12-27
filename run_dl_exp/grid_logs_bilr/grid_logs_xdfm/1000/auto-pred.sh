set -x

python ../pred.py --dataset_path ../../data/random/ --dataset_part trva --model_path ./xdfm_lr-0.001_l2-1e-06_bs-8192_trva.pt --model_name xdfm  
python ../cal_revenue.py tmp.pred.0 ../../data/random/gt.svm
python ../cal_revenue.py tmp.pred.3 ../../data/random/gt.svm
python ../cal_revenue.py tmp.pred.4 ../../data/random/gt.svm
python ../cal_revenue.py tmp.pred.5 ../../data/random/gt.svm
python ../cal_revenue.py tmp.pred.6 ../../data/random/gt.svm
mv tmp.pred.0 ./xdfm.pred.0
mv tmp.pred.3 ./xdfm.pred.3
mv tmp.pred.4 ./xdfm.pred.4
mv tmp.pred.5 ./xdfm.pred.5
mv tmp.pred.6 ./xdfm.pred.6

