#!/bin/bash

data='context.ffm'

random_data='rd.trva.ffm'
filter_data='filter.ffm'
truth_data='truth.ffm'

shuf ${data}  > ${data}.shuf
data=${data}.shuf

# 5%(pre model) 85%(D.ffm DR.ffm) 10%(gt.ffm)
total_num=`wc -l ${data} | awk '{print $1}'`
small_perc_total_num=`echo "scale=0;$total_num*5/100" | bc -l `
large_perc_total_num=`echo "scale=0;$total_num*85/100" | bc -l `
echo $total_num $small_perc_total_num $large_perc_total_num

head -n $small_perc_total_num ${data} > $random_data &
head -n $(($small_perc_total_num + $large_perc_total_num)) ${data} | tail -n $large_perc_total_num > $filter_data &
tail -n $(($total_num - $small_perc_total_num - $large_perc_total_num)) ${data} > $truth_data &

wait

# Transform random_data to binary data
python random_select.py ${random_data}

# Split random_data to tr va
half_small_perc_total_num=`echo "scale=0;$small_perc_total_num/2" | bc -l `
head -n $half_small_perc_total_num "${random_data}.bin" > rd.tr.ffm.bin &
tail -n $(($small_perc_total_num - $half_small_perc_total_num)) "${random_data}.bin" > rd.va.ffm.bin &

wait
