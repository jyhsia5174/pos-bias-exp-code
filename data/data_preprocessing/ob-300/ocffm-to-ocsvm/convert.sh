#! /bin/bash

ffm_ext=$1
svm_ext="${ffm_ext/ffm/svm}"

ffm_files="item.ffm det.${ffm_ext} random.${ffm_ext} 
greedy_random.${ffm_ext} random_greedy.${ffm_ext} truth.ffm"
svm_files="item.svm det.${svm_ext} random.${svm_ext} 
greedy_random.${svm_ext} random_greedy.${svm_ext} truth.svm"

# First file must be item.ffm 
python ocffm-to-ocsvm.py ${ffm_files}

#echo "tar --dereference -zcvf ob.300.${ext}.tar.gz item.ffm det.${ext} prop.${ext} random.${ext} truth.ffm item.svm det.${svm_ext} prop.${svm_ext} random.${svm_ext} truth.svm "
#
#tar --dereference -zcvf ob.300.${ext}.tar.gz item.ffm det.${ext} prop.${ext} random.${ext} truth.ffm item.svm det.${svm_ext} prop.${svm_ext} random.${svm_ext} truth.svm 
