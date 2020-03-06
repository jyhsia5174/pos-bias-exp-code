root=$1
mode=$2
model=$3

for i in 'der.0.01' 'der.0.1' 'der.comb.0.01' 'der.comb.0.1' 'derive.det' 'derive.random'
do
	python get_record.py ${root}/${i}.${model} ${mode} > ${i}.record
done

paste *.record | column -s $' ' -t
rm *.record

