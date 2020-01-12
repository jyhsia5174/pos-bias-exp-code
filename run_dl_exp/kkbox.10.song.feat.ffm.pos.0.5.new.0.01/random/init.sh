#!/bin/bash

d=`pwd | cut -d'/' -f6`
d2=`pwd | cut -d'/' -f7`
echo ${d}
ln -sf ../../../data/mix/${d}/*gt.svm ./
ln -sf ../../../data/mix/${d}/item.svm ./
ln -sf ../../../data/mix/${d}/truth.svm ./gt.svm
for i in 'trva' 'tr' 'va'
do
    ln -sf ../../../data/mix/${d}/select_${i}.svm ${i}.svm
done
