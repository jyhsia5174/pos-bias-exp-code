item='item.ffm'
tr5='rd.trva.ffm'
tr90='filter.ffm'
model='16.45.model'

$time 
echo "./predict-and-filter -c 5 --ns $item $tr5 $tr90 $model"
