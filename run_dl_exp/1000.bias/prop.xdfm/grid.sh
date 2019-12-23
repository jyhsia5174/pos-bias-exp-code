# Init
#source activate py3.7

# Train file path
train='python ../../main.py'

# Fixed parameter
flag='train'
epoch=25
batch_size=8192

# Data set
ds='pos'
ds_part='tr'
ds_path='./'

# Log path
log_path="grid_logs"

# others
mn='xdfm'
device='cuda:1'

task(){
# Set up fixed parameter and train command
train_cmd="${train} --dataset_name ${ds} --dataset_part ${ds_part} --dataset_path ${ds_path} --flag ${flag} --model_name ${mn} --epoch ${epoch} --device ${device} --save_dir ${log_path} --batch_size ${batch_size}"

# Print out all parameter pair
for lr in 0.0001 0.001
do
    for wd in 5e-6 1e-6 5e-7
    do
        for k in 16 32 
        do
            cmd=${train_cmd}
            cmd="${cmd} --learning_rate ${lr}"
            cmd="${cmd} --weight_decay ${wd}"
            cmd="${cmd} --embed_dim ${k}"
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
