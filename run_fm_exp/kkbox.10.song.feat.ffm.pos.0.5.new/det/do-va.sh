#! /bin/bash
. ./init.sh
#d=`pwd | cut -d'/' -f6`
ln -sf ../../../data/mix/kkbox.10.song.feat.ffm.pos.0.5.new/rnd_va.ffm ./

# Train file path
train='./hytrain'

# Fixed parameter
wn=1
c=8

# Data setr
tr='trva.ffm'
va='rnd_va.ffm'
item='item.ffm'

# Log path
log_path="va-score"
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
for i in '' '.const' '.pos'
do
	python ../../cal_auc.py ./ ../../../data/mix/kkbox.10.song.feat.ffm.pos.0.5.new/rnd_va.10${i}.ffm 10
done
mv Pva* Qva* ${log_path}
