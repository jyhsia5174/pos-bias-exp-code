#!/bin/bash

random_data='rd.ffm'
filter_data='fl.ffm'
truth_data='truth.ffm'

total_num=`wc -l ob_all_1000.ffm | awk '{print $1}'`
small_perc_total_num=`echo "scale=0;$total_num*5/100" | bc -l `
large_perc_total_num=`echo "scale=0;$total_num*90/100" | bc -l `
echo $total_num $small_perc_total_num $large_perc_total_num


head -n $small_perc_total_num ob_all_1000.ffm > $random_data &
head -n $(($small_perc_total_num + $large_perc_total_num)) ob_all_1000.ffm | tail -n $large_perc_total_num > $filter_data &
tail -n $(($total_num - $small_perc_total_num - $large_perc_total_num)) ob_all_1000.ffm > $truth_data &

wait

half_small_perc_total_num=`echo "scale=0;$small_perc_total_num/2" | bc -l `
head -n $half_small_perc_total_num $random_data > rd.tr.ffm &
tail -n $(($small_perc_total_num - $half_small_perc_total_num)) $random_data > rd.va.ffm &
