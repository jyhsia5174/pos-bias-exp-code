#! /bin/bash

check(){
    echo $1
    for i in {1..10}
    do
        awk '{print $1}' $1 | awk -F , -v pos=$i '{print $pos}' | grep ':1:' | wc -l
    done

}


check det.ffm.pos.bias 
check det.ffm 
check prop.ffm.pos.bias
check prop.ffm
check random.ffm.pos.bias
check random.ffm
