set -x

python main.py --dataset_path ../data/random-top10/ --dataset_part trva --model_name dssm --save_dir top10
python main.py --dataset_path ../data/random-top10/ --dataset_part trva-nopos --model_name dssm --save_dir top10
python main.py --dataset_path ../data/random-top100/ --dataset_part trva --model_name dssm --save_dir top100
python main.py --dataset_path ../data/random-top100/ --dataset_part trva-nopos --model_name dssm --save_dir top100
#python main.py --dataset_path ../data/random/ --model_name bidssm
#python main.py --dataset_path ../data/random/ --model_name extdssm
