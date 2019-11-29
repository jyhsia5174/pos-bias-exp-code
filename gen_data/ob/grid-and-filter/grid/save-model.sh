
t='45'
l='16'

./train -l ${l} -t ${t} -w 1 -r -1 -c 5 -k 8 -o ${l}.${t}.model --ns item.ffm rd.trva.ffm
