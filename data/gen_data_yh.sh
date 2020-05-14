#!/bin/bash

yh_remap_dir=$1

mkdir yahooR3
cd yahooR3
ln -sf $PWD/../$1 origin
mkdir derive
cd derive

ln -sf ../origin/item item.ffm
ln -sf ../origin/va.1.remap random_va.ffm
ln -sf ../origin/va.1.remap det_va.ffm
ln -sf ../origin/te.1.remap rnd_gt.ffm
ln -sf ../../scripts/ocffm-to-ocsvm.py ./
ln -sf ../../scripts/ctr.py ./

ln -sf ../origin/tr.1.remap random_tr.ffm
ln -sf ../origin/trva.1.remap random_trva.ffm
ln -sf ../origin/tr.99.remap det_tr.ffm
ln -sf ../origin/trva.99.remap det_trva.ffm
ln -sf ../origin/tr.100.remap select_tr.ffm
ln -sf ../origin/trva.100.remap select_trva.ffm

python ocffm-to-ocsvm.py item.ffm det_tr.ffm det_trva.ffm det_va.ffm select_tr.ffm select_trva.ffm random_tr.ffm random_trva.ffm random_va.ffm rnd_gt.ffm 
python ctr.py

cd ../
mkdir der.comb.0.01
cd der.comb.0.01

for i in 'ffm' 'svm'
do
	for j in 'tr' 'trva'
	do
		ln -sf ../derive/select_$j.$i ./
	done
	ln -sf ../derive/random_va.$i select_va.$i
	ln -sf ../derive/rnd_gt.$i ./
done
ln -sf ../derive/item.* ./

cd ../
mkdir der.comb.0.01.obs
cd der.comb.0.01.obs
ln -sf ../../scripts/gen_obs.py ./
ln -sf ../derive/item.svm ./
ln -sf obs_va.svm rnd_gt.svm
python gen_obs.py ../derive

cd ../../
