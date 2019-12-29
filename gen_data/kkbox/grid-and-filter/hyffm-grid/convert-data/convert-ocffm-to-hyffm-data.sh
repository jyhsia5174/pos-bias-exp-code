#!/bin/bash

awk '{printf "%s:1", $1; $1=""; print $0}' $1 > $1.cvt

