#! /bin/bash
. ./init.sh

# Train file path
train='./hytrain'

# Fixed parameter
wn=1
c=6

# Data setr
tr='trva.ffm'
va='rnd_gt.ffm'
item='item.ffm'

# Log path
log_path="test-score"
mkdir -p $log_path


task(){

# Print out all parameter pair
t=`python ../../select_params.py logs logloss | cut -d' ' -f3`
l=`python ../../select_params.py logs logloss | cut -d' ' -f1`
k=`python ../../select_params.py logs logloss | cut -d' ' -f2`
w=0
r=-1
train_cmd="${train} -wn ${wn} -k ${k} -c ${c} --ns"
cmd=${train_cmd}
cmd="${cmd} -l ${l}"
cmd="${cmd} -w ${w}"
cmd="${cmd} -r ${r}"
cmd="${cmd} -t ${t}"
echo "${cmd} -p ${va} ${item} ${tr} > ${log_path}/$l.$w.$k.log"
}


# Check command
echo "Check command list (the command may not be runned!!)"
task
wait


# Run
echo "Run"
task | xargs -0 -d '\n' -P 3 -I {} sh -c {} 
