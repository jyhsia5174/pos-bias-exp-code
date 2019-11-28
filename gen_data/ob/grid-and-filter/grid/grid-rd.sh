train='./train'
t=100
w=1
r='-1'
c=5
k=8
others=''

va='rd.va.ffm'
tr='rd.tr.ffm'
item='item.ffm'

logs_path="logs/$tr.$va.$item.$t.$w.$r.$k"
mkdir -p $logs_path

task(){
    for l in 1 4 16
    do
        echo "$train -l $l -t $t -p $va -w $w -r $r -c $c -k $k $others $item $tr > $logs_path/$l.log"
    done
}

task
#task | xargs -0 -d '\n'  -P 4 -I {} sh -c {} &
