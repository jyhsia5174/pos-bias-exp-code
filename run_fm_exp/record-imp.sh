
root=$1
mode=$2
cat ${root}/${mode}.record | awk '{if (NF==2 && substr($0,1,1)=='0') print $0}'
