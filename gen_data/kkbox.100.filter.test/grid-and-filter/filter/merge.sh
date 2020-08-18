# Config
R='R.label'
D='D.label'
RD='RD.label'
DR='DR.label'
filter_data='filter.ffm'

awk '{$1=""; print $0}' $filter_data > $filter_data.tmp &

sed -i 's/,$//g' $R 
sed -i 's/,$//g' $D 
sed -i 's/,$//g' $RD 
sed -i 's/,$//g' $DR 

paste $R $filter_data.tmp > R.ffm & 
paste $D $filter_data.tmp > D.ffm & 
paste $RD $filter_data.tmp > RD.ffm &
paste $DR $filter_data.tmp > DR.ffm &

wait
rm $filter_data.tmp

