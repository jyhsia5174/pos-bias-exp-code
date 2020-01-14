# Init
#source activate py3.7

# params 
gpu=$1
mode=$2
model_name=$3

# Data set
ds='pos'
tr_part='tr'
va_part='va'
ds_path='./'

# Fixed parameter
flag='train'
epoch=50
bs=128

# others
log_path="logs"
device="cuda:${gpu}"

task(){
# Set up fixed parameter and train command
cmd="python ../../main.py"
cmd="${cmd} --dataset_name ${ds}"
cmd="${cmd} --train_part ${tr_part}"
cmd="${cmd} --valid_part ${va_part}"
cmd="${cmd} --dataset_path ${ds_path}"
cmd="${cmd} --flag ${flag}"
cmd="${cmd} --model_name ${model_name}"
cmd="${cmd} --epoch ${epoch}"
cmd="${cmd} --device ${device}"
cmd="${cmd} --save_dir ${log_path}"
cmd="${cmd} --batch_size ${bs}"

# Print out all parameter pair
for lr in 0.0001 #0.001
do
    for wd in 1e-2 #1e-4 1e-6 
    do
        for k in 16 #32 64 
        do
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
