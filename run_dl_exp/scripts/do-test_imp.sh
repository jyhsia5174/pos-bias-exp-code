# Init
#source activate py3.7

# params 
gpu=$1
mode=$2
model_name=$3
ps=$4
imp_dir=$5

root=`pwd`
if [ -d "${imp_dir}" ]; then
	cd ${imp_dir}
	imp_lr=`python select_params.py logs ${mode} | cut -d' ' -f1`
	imp_wd=`python select_params.py logs ${mode} | cut -d' ' -f2`
	imp_bs=`python select_params.py logs ${mode} | cut -d' ' -f3` 
	imp_k=`python select_params.py logs ${mode} | cut -d' ' -f4`
	imp_epoch=`python select_params.py logs ${mode} | cut -d' ' -f5`
	imp_epoch=$((${epoch}+1))
	cd ${root}
else
	echo "Can not find dir ${imp_dir}!"
	exit 0
fi

# Data set
ds='pos'
tr_part='sc_trva'
va_part='rnd_gt'
ds_path='./'

# Fixed parameter
flag='imp_train'
lr=`python select_params_imp.py logs ${mode} | cut -d' ' -f1`
wd=`python select_params_imp.py logs ${mode} | cut -d' ' -f2`
bs=`python select_params_imp.py logs ${mode} | cut -d' ' -f3` 
k=`python select_params_imp.py logs ${mode} | cut -d' ' -f4`
omega=`python select_params_imp.py logs ${mode} | cut -d' ' -f5`
epoch=`python select_params_imp.py logs ${mode} | cut -d' ' -f6`
epoch=$((${epoch}+1))

# others 
log_path="test-score.${mode}"
device="cuda:${gpu}"

# imp model training
imp_task(){
# Set up fixed parameter and train command
cmd="python ../../main.py"
cmd="${cmd} --dataset_name ${ds}"
cmd="${cmd} --train_part st_trva"
cmd="${cmd} --valid_part ${va_part}"
cmd="${cmd} --dataset_path ${ds_path}"
cmd="${cmd} --flag train"
cmd="${cmd} --model_name ${model_name}"
cmd="${cmd} --epoch ${imp_epoch}"
cmd="${cmd} --device ${device}"
cmd="${cmd} --save_dir imp_log_trva"
cmd="${cmd} --batch_size ${imp_bs}"
cmd="${cmd} --ps ${ps}"
cmd="${cmd} --learning_rate ${imp_lr}"
cmd="${cmd} --weight_decay ${imp_wd}"
cmd="${cmd} --embed_dim ${imp_k}"
echo "${cmd}"
}
# Run
echo "Train imp model"
imp_task
imp_task | xargs -0 -d '\n' -P 1 -I {} sh -c {}

imp_model_path=`find ./imp_log_trva -name "*.pt"`
imp_model_name=`echo $(basename ${model_path}) | cut -d'_' -f1`
echo "imp_model_name: ${imp_model_name}, imp_model_path: ${imp_model_path}"

task(){
# Set up fixed parameter and train command
cmd="python ../../main.py"
cmd="${cmd} --dataset_name ${ds}"
cmd="${cmd} --train_part ${tr_part}"
cmd="${cmd} --valid_part ${va_part}"
cmd="${cmd} --dataset_path ${ds_path}"
cmd="${cmd} --flag ${flag}"
cmd="${cmd} --model_name ${model_name}"
cmd="${cmd} --imp_model_path ${imp_model_path}"
cmd="${cmd} --epoch ${epoch}"
cmd="${cmd} --device ${device}"
cmd="${cmd} --save_dir ${log_path}"
cmd="${cmd} --batch_size ${bs}"
cmd="${cmd} --ps ${ps}"
cmd="${cmd} --learning_rate ${lr}"
cmd="${cmd} --weight_decay ${wd}"
cmd="${cmd} --embed_dim ${k}"
cmd="${cmd} --omega ${omega}"
echo "${cmd}"
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
#wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
