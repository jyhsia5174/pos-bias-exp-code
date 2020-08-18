#! /bin/bash

# context.ffm item.ffm cp to split-data/

# linux13:/tmp2/b02701216/pos-bias-exp/data-gen/kkbox-100-more-context-no-filter-no-bid/gen_data/kkbox-100/kkbox_csv_to_ocffm

# 100 --> 90 5 5
# 90 1pos --> random 10 --> binary data
cd split-data
./split.sh

# save 90 model
cd ../grid-and-filter/hyffm-grid
source init.sh; make
./save_model.sh

# D.ffm DR.ffm RD.ffm
cd ../filter
source init.sh; make
./run-filter.sh

# Add 0.5 bias
cd ../../add_bias
./add_bias_all.sh
