#! /bin/bash

ext=$1

svm_ext="${ext/ffm/svm}"

python ocffm-to-ocsvm.py item.ffm det.${ext} prop.${ext} random.${ext} truth.ffm

echo "tar --dereference -zcvf ob.300.${ext}.tar.gz item.ffm det.${ext} prop.${ext} random.${ext} truth.ffm item.svm det.${svm_ext} prop.${svm_ext} random.${svm_ext} truth.svm "

tar --dereference -zcvf ob.300.${ext}.tar.gz item.ffm det.${ext} prop.${ext} random.${ext} truth.ffm item.svm det.${svm_ext} prop.${svm_ext} random.${svm_ext} truth.svm 
