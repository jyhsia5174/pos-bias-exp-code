set -x

python ../../main.py --dataset_path ./ --dataset_part trva --model_path test_score/dssm_lr-0.0001_l2-1e-06_bs-8192_trva.pt --model_name dssm --flag test_auc --batch_size 105500 --device cuda:0

