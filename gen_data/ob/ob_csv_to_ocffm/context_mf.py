#! /usr/bin/python3
from __future__ import print_function
import csv

# Field mf
keys1 = ['uuid']

feat_dict = {}
incr_num1 = -1


def add_feat(key, value, field):
    global incr_num1
    global feat_dict
    real_key = "{0}:{1}".format(key, value)
    if real_key in feat_dict:
        return feat_dict[real_key]
    incr_num1 += 1
    feat_idx = incr_num1
    feat_dict[real_key] = feat_idx
    return feat_idx

def make_tuple(feat_list,field):
    feat_str = ["%d:1" % i for i in feat_list]
    fnc = lambda x: "{}:{}".format(int(field), x)
    return list(map(fnc, feat_str))

def convert2ffm( o_f, i_f ):
    svm_f = open(o_f, 'w')
    for line in csv.DictReader(open(i_f), delimiter=','):
        feat_idx_list = []
        output = line['label']
        for key in keys1:
            if line[key] == "":
                continue
            else:
                feat_idx_list.append(add_feat(key, line[key], 0))
        output = "{} {}".format(output," ".join(make_tuple(feat_idx_list,0)))
        print(output,file=svm_f)

if __name__ == '__main__':
    convert2ffm('ob.tr.mf', 'ob.tr.csv')
    convert2ffm('ob.va.mf', 'ob.va.csv')
    convert2ffm('ob.te.mf', 'ob.te.csv')

