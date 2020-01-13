#! /bin/bash
. ./init.sh

# Train file path
train='./hytrain'

# params selection
mode=$1

# Fixed parameter
wn=1
c=12

# Data setr
tr='trva.ffm'
va='rnd_gt.ffm'
item='item.ffm'

# Log path
log_path="test-score.${mode}"
mkdir -p $log_path


task(){

# Print out all parameter pair
echo "python select_params.py logs ${mode} | cut -d' ' -f3"
t=`python select_params.py logs ${mode} | cut -d' ' -f3`
l=`python select_params.py logs ${mode} | cut -d' ' -f1`
k=`python select_params.py logs ${mode} | cut -d' ' -f2`
w=0
r=-1
train_cmd="${train} -wn ${wn} -k ${k} -c ${c} --ns"
cmd=${train_cmd}
cmd="${cmd} -l ${l}"
cmd="${cmd} -w ${w}"
cmd="${cmd} -r ${r}"
cmd="${cmd} -t ${t}"
cmd="${cmd} -p ${va} ${item} ${tr} > ${log_path}/$l.$w.$k.log"
echo "${cmd}; mv Pva* Qva* ${log_path}"
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 1 -I {} sh -c {} 
