# Init
#source activate py3.7
. ./init.sh

# Train file path
train='python ../../main.py'


# Data set
ds='pos'
ds_part='tr'
ds_path='./'

# Log path
log_path="grid_logs"

# Fixed parameter
device="cuda:$1"
mn=$2
flag='train'
epoch=30
#k=32
batch_size=128

task(){
# Set up fixed parameter and train command
train_cmd="${train} --dataset_name ${ds} --dataset_part ${ds_part} --dataset_path ${ds_path} --flag ${flag} --model_name ${mn} --epoch ${epoch} --device ${device} --save_dir ${log_path} "

# Print out all parameter pair
for lr in 0.0001 0.001
do
    for wd in 1e-2 1e-4 1e-6
    do
        for k in 64 16 32 
        #for batch_size in 128 512 2048
        do
            cmd=${train_cmd}
            cmd="${cmd} --learning_rate ${lr}"
            cmd="${cmd} --weight_decay ${wd}"
            cmd="${cmd} --embed_dim ${k}"
			cmd="${cmd} --batch_size ${batch_size}"
            echo "${cmd}"
        done
    done
done
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
#wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {}
