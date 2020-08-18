#! /bin/bash

# link context.ffm item.ffm to ob_csv_to_ocffm

# change 5% 85%
# file location: split-data/split.sh
    ## 5%(pre model) 85%(D.ffm DR.ffm) 10%(gt.ffm)
    #total_num=`wc -l ${data} | awk '{print $1}'`
    #small_perc_total_num=`echo "scale=0;$total_num*5/100" | bc -l `
    #large_perc_total_num=`echo "scale=0;$total_num*85/100" | bc -l `
    #echo $total_num $small_perc_total_num $large_perc_total_num

# rd.*.ffm.bin (5%) filter.ffm(85% D.ffm DR.ffm) truth.ffm(10% gt.ffm)
cd split-data
./split.sh

# Train pre model (filter model)
cd ../grid-and-filter/hyffm-grid
./grid.sh; ./save_model.sh

# Filter filter.ffm --> D.ffm ...
cd ../filter
./run-filter.sh

# Add bias
cd ../../add_bias
./add_bias_all.sh
