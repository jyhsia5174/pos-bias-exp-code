#!/bin/bash

source init.sh
make

c=5
k=8
t=20
tr=rd.tr.ffm.bin
va=rd.va.ffm.bin
item=item.ffm
logs_pth=logs/

mkdir -p $logs_pth

# 2^0 ~ 2^-11
w_train=(0)
wn_train=(1)
l_train=(6.25e-2 0.25 1 4 16)
r_train=(0)

task(){
  for w in ${w_train[@]} 
  do
      for r in ${r_train[@]}
      do
        for l in ${l_train[@]}
        do
          echo "./train -k $k -l $l -t ${t} -r $r -w $w -wn 1 -c ${c} -p ${va} ${item} ${tr} > $logs_pth/$l"
        done
      done
  done
}


num_core=4

#task
task | xargs -d '\n' -P $num_core -I {} sh -c {} 
