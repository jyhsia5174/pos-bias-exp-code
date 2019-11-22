#! /usr/bin/python3
PATH = 'ad_filter.csv'
import csv

# No Field
keys1 = ['ad_id']

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

if __name__ == '__main__':
    item_svm = open('item.mf', 'w')
    for line in csv.DictReader(open(PATH), delimiter=','):
        feat_idx_list = []
        for key in keys1:
            if line[key] == "":
                continue
            feat_idx_list.append(add_feat(key, line[key].strip(), 0))
        output = "{}".format(" ".join(make_tuple(feat_idx_list, 0)))
        print( output, file=item_svm )

