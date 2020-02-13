#! /bin/bash

item='item.ffm'
tr5='rd.trva.ffm'
tr90='test.ffm'
model='filter.model'

$time 
echo "./predict-and-filter -c 1 $item $tr5 $tr90 $model"


source init.sh
./predict-and-filter -c 1 $item $tr5 $tr90 $model

