# Config
rd='random_filter'
pr='propensious_filter'
de='determined_filter'

filter_data='trvate.90.ffm'
awk '{$1=""; print $0}' $filter_data > $filter_data.tmp &

paste $rd.pos.bias $filter_data.tmp > random.pos.ffm & 
paste $pr.pos.bias $filter_data.tmp > prop.pos.ffm &
paste $de.pos.bias $filter_data.tmp > det.pos.ffm &

