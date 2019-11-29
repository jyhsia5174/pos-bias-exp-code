
t='55'
l='16'

./train -l ${l} -t ${t} -w 0 -r -1 -c 5 -k 8 -o ${l}.${t}.model --ns item.ffm rd.trva.ffm
