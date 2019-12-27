# Init
#source activate py3.7

# Train file path
train='python ../../main.py'

# Fixed parameter
flag='train'
epoch=30
batch_size=8192

# Data set
ds='pos'
ds_part='trva'
ds_path='./'

# Log path
log_path="test_score"

# others
mn='dssm'
device='cuda:1'

task(){
# Set up fixed parameter and train command
train_cmd="${train} --dataset_name ${ds} --dataset_part ${ds_part} --dataset_path ${ds_path} --flag ${flag} --model_name ${mn} --epoch ${epoch} --device ${device} --save_dir ${log_path}"

# Print out all parameter pair
lr=0.0001
wd=1e-6
bs=8192 
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
