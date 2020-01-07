# Init
#source activate py3.7

# Train file path
train='python ../../main.py'

# Fixed parameter
flag='train'
epoch=`python ../../select_params.py grid_logs auc | cut -d' ' -f5`
k=`python ../../select_params.py grid_logs auc | cut -d' ' -f4`

# Data set
ds='pos'
ds_part='trva'
va_part='rnd_gt'
ds_path='./'

# Log path
log_path="test_score"

# others
device="cuda:$1"
mn=$2

task(){
# Set up fixed parameter and train command
train_cmd="${train} --dataset_name ${ds} --dataset_part ${ds_part} --valid_part ${va_part} --dataset_path ${ds_path} --flag ${flag} --model_name ${mn} --epoch ${epoch} --device ${device} --save_dir ${log_path} --embed_dim ${k}"

# Print out all parameter pair
lr=`python ../../select_params.py grid_logs auc | cut -d' ' -f1`
wd=`python ../../select_params.py grid_logs auc | cut -d' ' -f2`
bs=`python ../../select_params.py grid_logs auc | cut -d' ' -f3` 
cmd=${train_cmd}
cmd="${cmd} --learning_rate ${lr}"
cmd="${cmd} --weight_decay ${wd}"
cmd="${cmd} --batch_size ${bs}"
echo "${cmd}"
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
#wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
