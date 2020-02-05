#! /bin/bash
. ./init.sh

# Train file path
train='./hytrain'

# Fixed parameter
t=50
c=8

# Data set
tr='tr.ffm'
va='va.ffm'
item='item.ffm'

# Log path
log_path="logs"
mkdir -p $log_path


task(){
# Set up fixed parameter and train command
train_cmd="${train} -t ${t} -c ${c} "

# Print out all parameter pair
for l in 625e-4 25e-2 1 4 16 32 #64 #128
do
    for w in 0 
    do
        for d in 32
        do
            cmd=${train_cmd}
            cmd="${cmd} -l ${l}"
            #cmd="${cmd} -w ${w}"
            cmd="${cmd} -d ${d}"
            echo "${cmd} -p ${va} ${item} ${tr} > ${log_path}/$l.$w.$d.log"
        done
    done
done
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 3 -I {} sh -c {} 
