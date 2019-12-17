set -x
#python pred.py --dataset_path ../data/random/ --model_path tmp/extdssm_lr-0.001_l2-1e-06_bs-8192.pt --model_name extdssm
#python cal_revenue.py tmp.pred.0 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.3 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.4 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.5 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.6 ../data/random/gt.svm
#mv tmp.pred.0 extdssm.pred.0
#mv tmp.pred.3 extdssm.pred.3
#mv tmp.pred.4 extdssm.pred.4
#mv tmp.pred.5 extdssm.pred.5
#mv tmp.pred.6 extdssm.pred.6
#
#python pred.py --dataset_path ../data/random/ --model_path tmp/bidssm_lr-0.001_l2-1e-06_bs-8192.pt --model_name bidssm
#python cal_revenue.py tmp.pred.0 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.3 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.4 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.5 ../data/random/gt.svm
#python cal_revenue.py tmp.pred.6 ../data/random/gt.svm
#mv tmp.pred.0 bidssm.pred.0
#mv tmp.pred.3 bidssm.pred.3
#mv tmp.pred.4 bidssm.pred.4
#mv tmp.pred.5 bidssm.pred.5
#mv tmp.pred.6 bidssm.pred.6

python pred.py --dataset_path ../data/random-top10/ --dataset_part trva --model_path top10/dssm_lr-0.001_l2-1e-06_bs-8192_trva.pt --model_name dssm
python cal_revenue.py tmp.pred.0 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.3 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.4 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.5 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.6 ../data/random-top10/gt.svm
mv tmp.pred.0 top10/dssm.pred.0
mv tmp.pred.3 top10/dssm.pred.3
mv tmp.pred.4 top10/dssm.pred.4
mv tmp.pred.5 top10/dssm.pred.5
mv tmp.pred.6 top10/dssm.pred.6

python pred.py --dataset_path ../data/random-top10/ --dataset_part trva --model_path top10/dssm_lr-0.001_l2-1e-06_bs-8192_trva-nopos.pt --model_name dssm
python cal_revenue.py tmp.pred.0 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.3 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.4 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.5 ../data/random-top10/gt.svm
python cal_revenue.py tmp.pred.6 ../data/random-top10/gt.svm
mv tmp.pred.0 top10/dssm-nopos.pred.0
mv tmp.pred.3 top10/dssm-nopos.pred.3
mv tmp.pred.4 top10/dssm-nopos.pred.4
mv tmp.pred.5 top10/dssm-nopos.pred.5
mv tmp.pred.6 top10/dssm-nopos.pred.6

python pred.py --dataset_path ../data/random-top100/ --dataset_part trva --model_path top100/dssm_lr-0.001_l2-1e-06_bs-8192_trva.pt --model_name dssm
python cal_revenue.py tmp.pred.0 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.3 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.4 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.5 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.6 ../data/random-top100/gt.svm
mv tmp.pred.0 top100/dssm.pred.0
mv tmp.pred.3 top100/dssm.pred.3
mv tmp.pred.4 top100/dssm.pred.4
mv tmp.pred.5 top100/dssm.pred.5
mv tmp.pred.6 top100/dssm.pred.6

python pred.py --dataset_path ../data/random-top100/ --dataset_part trva --model_path top100/dssm_lr-0.001_l2-1e-06_bs-8192_trva-nopos.pt --model_name dssm
python cal_revenue.py tmp.pred.0 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.3 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.4 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.5 ../data/random-top100/gt.svm
python cal_revenue.py tmp.pred.6 ../data/random-top100/gt.svm
mv tmp.pred.0 top100/dssm-nopos.pred.0
mv tmp.pred.3 top100/dssm-nopos.pred.3
mv tmp.pred.4 top100/dssm-nopos.pred.4
mv tmp.pred.5 top100/dssm-nopos.pred.5
mv tmp.pred.6 top100/dssm-nopos.pred.6

