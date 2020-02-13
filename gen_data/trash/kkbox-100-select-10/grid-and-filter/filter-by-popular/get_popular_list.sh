#! /usr/bin/bash

data=$1

awk '{print $1}' ${data} | sort | uniq -c | sort -grk 1 | awk '{print $2}'  | awk -F ':' '{print $1}' > popular_rank
