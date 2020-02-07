#! /bin/bash

ext=$1
name=$2

if [ -z ${ext} ] || [ -z ${name} ]
then
    echo "specify ext & name"
    exit
fi
echo "ext: \"${ext}\" name: \"${name}\""

svm_ext="${ext/ffm/svm}"

python ocffm-to-ocsvm.py item.ffm det.${ext} truth.ffm

cmd="tar --dereference -zcvf ${name}.${ext}.tar.gz item.ffm det.${ext} truth.ffm item.svm det.${svm_ext} truth.svm "
echo $cmd | xargs -I {} sh -c {} 
