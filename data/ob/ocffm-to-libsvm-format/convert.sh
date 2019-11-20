item=item.ffm
tr=tr.ffm
va=va.ffm
gt=gt.ffm

python ocffm-to-libsvm.py $item $tr $va $gt
#python ocffm-to-libsvm.py $item $gt
