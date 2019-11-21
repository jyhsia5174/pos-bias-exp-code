item='item.ffm'
#tr5='rd.ffm'
tr5='trva.rd.ffm'
tr90='fl.ffm'
model='16.model'

time ./predict-and-filter -c 5 $item $tr5 $tr90 $model
