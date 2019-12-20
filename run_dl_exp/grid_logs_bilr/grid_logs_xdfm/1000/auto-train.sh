set -x

python ../main.py --dataset_part trva --dataset_path ../../data/random/ --model_name xdfm --epoch 30 --save_dir ./ --weight_decay 1e-6 
