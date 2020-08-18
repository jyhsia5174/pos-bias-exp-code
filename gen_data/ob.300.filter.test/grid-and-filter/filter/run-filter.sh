#! /bin/bash

item='item.ffm'
tr5='rd.trva.ffm.bin'
tr90='filter.ffm'
model='filter.model'

$time 
echo "./predict-and-filter -c 5 $item $tr5 $tr90 $model"


source init.sh
make
./predict-and-filter -c 5 $item $tr5 $tr90 $model

./merge.sh
