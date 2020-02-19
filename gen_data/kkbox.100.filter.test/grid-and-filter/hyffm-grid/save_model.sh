#!/bin/bash

source init.sh

# Fixed param
r=0
w=0
tr='rd.trva.ffm.bin'

# va logs
logs='logs'
cd ${logs}
l=`cat * | sed '/iter/d' | sort -gk 3 | tail -n1  | xargs -I {}  grep --files-with-matches '{}' * `
t=` cat * | sed '/iter/d' | sort -gk 3 | tail -n1  | awk '{print $1}' `
echo $l $t
cd ..

cmd="./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 10 -o filter.model item.ffm ${tr} > save_model.log"

echo "${cmd}"
#echo "${cmd}" > filter.model.param
#echo "${cmd}" | xargs -I {} sh -c {} 
