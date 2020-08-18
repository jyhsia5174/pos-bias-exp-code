#!/bin/bash

source init.sh

tr='filter.ffm.rnd.select'
va='filter.ffm.rnd.select'

l=0.1
r=0
w=0

t=20


cmd="./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 10 -p ${va} -o filter.model item.ffm ${tr} > save_model.log"

echo "${cmd}"
echo "${cmd}" > filter.model.param
echo "${cmd}" | xargs -I {} sh -c {} &
