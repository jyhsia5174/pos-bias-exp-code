#! /bin/bash

check(){
    echo $1
    for i in {1..10}
    do
        awk -F , -v pos=$i '{print $pos}' $1 | grep ':1:' | wc -l
    done

}

check determined_filter.label
