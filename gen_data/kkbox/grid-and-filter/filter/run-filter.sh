item='item.ffm'
tr5='rd.trva.ffm'
tr90='filter.ffm'
model='rd.ffm.cvt.4.-6.0.0009765625.1.ffm-ffm.model'

$time 
echo "./predict-and-filter -c 5 $item $tr5 $tr90 $model"