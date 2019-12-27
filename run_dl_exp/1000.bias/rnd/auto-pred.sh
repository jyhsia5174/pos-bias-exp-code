set -x

python ../../main.py --dataset_path ./ --dataset_part trva --model_path test_score/dssm_lr-0.0001_l2-1e-06_bs-8192_trva.pt --model_name dssm --flag test --batch_size 105500 --device cuda:1
python ../../cal_revenue.py tmp.pred.0 ./gt.svm
python ../../cal_revenue.py tmp.pred.1 ./gt.svm
python ../../cal_revenue.py tmp.pred.2 ./gt.svm
python ../../cal_revenue.py tmp.pred.3 ./gt.svm
python ../../cal_revenue.py tmp.pred.4 ./gt.svm
mv tmp.pred.0 ./test_score/test.pred.0
mv tmp.pred.1 ./test_score/test.pred.1
mv tmp.pred.2 ./test_score/test.pred.2
mv tmp.pred.3 ./test_score/test.pred.3
mv tmp.pred.4 ./test_score/test.pred.4

