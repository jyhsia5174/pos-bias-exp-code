#! /bin/bash

if [ -z $1 ]
then
echo "Usage: ./split.sh file
This will give top 80% to tr, 80~90 to va and 90 to te.
Good luck!!
"
exit
fi

l=`wc -l < $1`
echo "$l"
sz=`echo "scale=0; $l / 10" | bc`
dsz=`echo "scale=0; 2 * $sz" | bc`
t_sz=`echo "scale=0; $l - $dsz" | bc `

echo "$sz $dsz $t_sz"

name=${1%\.*}
ext=${1##*\.}
echo "$name $ext"

head -n $t_sz $1 > ${name}.tr.${ext}
tail -n $dsz $1 | head -n $sz > ${name}.va.${ext}
tail -n $sz $1 > ${name}.te.${ext}


