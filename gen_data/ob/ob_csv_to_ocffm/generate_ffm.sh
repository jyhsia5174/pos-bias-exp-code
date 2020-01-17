#! /bin/bash

python3 filter.py
python3 add_label.py
python3 context_ffm.py
python3 item_ffm.py
awk '{$1=$1":1:1"; print $0}' ob_all.ffm > context.ffm
