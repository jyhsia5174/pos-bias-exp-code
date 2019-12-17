#!/bin/bash

l=1
t=2
r=-8
w=0.00390625

./train -k 8 -l $l -t $t -r $r -w $w -wn 1  -c 5 -o rd.ffm.cvt.$l.$r.$w.1.ffm-ffm.model item.ffm rd.trva.ffm.cvt & 
