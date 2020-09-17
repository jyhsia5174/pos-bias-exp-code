root=`pwd`

cd $1
mkdir RD
mkdir DR
mkdir cbias
mkdir normal
mv random_greedy.* RD
mv greedy_random.* DR
mv *unif* cbias
cp truth.* item.* RD
cp truth.* item.* DR
cp truth.* item.* cbias
mv det.* random.* item.* truth.* normal

cd RD
ln -sf random_greedy.ffm.pos.0.5.bias det.ffm.pos.0.5.bias
ln -sf random_greedy.svm.pos.0.5.bias det.svm.pos.0.5.bias
cd ../

cd DR
ln -sf greedy_random.svm.pos.0.5.bias det.svm.pos.0.5.bias
ln -sf greedy_random.ffm.pos.0.5.bias det.ffm.pos.0.5.bias
cd ../

cd cbias
ln -sf random.ffm.pos.0.5.unif.bias det.ffm.pos.0.5.bias
ln -sf random.svm.pos.0.5.unif.bias det.svm.pos.0.5.bias
cd ../

cd $root 
./gen_data_selfva.sh $1/normal 0.5
./gen_data_select_selfva.sh $1/RD 0.5
./gen_data_select_selfva.sh $1/DR 0.5
./gen_data_select_selfva.sh $1/cbias 0.5
