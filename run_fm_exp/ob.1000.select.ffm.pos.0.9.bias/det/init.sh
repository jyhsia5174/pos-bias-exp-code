#!/bin/bash
machine_name=`hostname`
if [ "$machine_name" = "peanuts" ]; then
  echo "$machine_name"
  source /home/skypole/intel/mkl/bin/mklvars.sh intel64
elif [ "$machine_name" = "optima" ]; then
  echo "$machine_name"
  source /home/ybw/intel/mkl/bin/mklvars.sh intel64
elif [[ "$machine_name" =~ linux[0-9]* ]]; then
  echo "$machine_name"
  source /tmp2/b02701216/intel/mkl/bin/mklvars.sh intel64
else
  echo "Not found"
fi
MKLPATH=$MKLROOT/lib/intel64_lin
MKLINCLUDE=$MKLROOT/include
echo "Init complete"
echo $MKLROOT
echo $MKLPATH
echo $MKLINCLUDE

ln -sf ../../hybrid-ocffm/train hytrain
d=`pwd | cut -d'/' -f6`
prefix=`pwd | cut -d'/' -f7`
echo ${d}
ln -sf ../../../data/${d}/*gt.ffm ./
ln -sf ../../../data/${d}/item.ffm ./
for i in 'trva' 'tr' 'va'
do
	ln -sf ../../../data/${d}/${prefix}_${i}.ffm ${i}.ffm
done


