# Config
rd='random_filter.label'
pr='propensious_filter.label'
de='determined_filter.label'
filter_data='filter.ffm'

awk '{$1=""; print $0}' $filter_data > $filter_data.tmp &

sed -i 's/,$//g' $rd 
sed -i 's/,$//g' $pr 
sed -i 's/,$//g' $de 

paste $rd $filter_data.tmp > random.ffm & 
paste $pr $filter_data.tmp > prop.ffm &
paste $de $filter_data.tmp > det.ffm &

wait
rm $filter_data.tmp

