#! /usr/bin/python3
PATH = 'ad_filter.csv'
import csv

# Field 1, 2
keys1 = ['source_id', 'publisher_id', 'document_id']
keys2 = ['campaign_id', 'advertiser_id']

feat_dict = {}
incr_num = -1


def add_feat(key, value, field):
    global incr_num
    global feat_dict
    real_key = "{0}:{1}".format(key, value)
    if real_key in feat_dict:
        return feat_dict[real_key]
    incr_num += 1
    feat_dict[real_key] = incr_num
    return incr_num


def make_tuple(feat_list,field):
    feat_str = ["%d:1" % i for i in feat_list]
    fnc = lambda x: "{}:{}".format(int(field), x)
    return list(map(fnc, feat_str))

item_svm = open('item.fm', 'w')
for line in csv.DictReader(open(PATH), delimiter=','):
    # Key1
    feat_idx_list = []
    for key in keys1:
        if line[key] == "":
            continue
        feat_idx_list.append(add_feat(key, line[key].strip(), 0))
    output = "{}".format(" ".join(make_tuple(feat_idx_list, 0)))
    
    # Key2
    feat_idx_list = []
    for key in keys2:
        if line[key] == "":
            continue
        feat_idx_list.append(add_feat(key, line[key].strip(), 0))
    output = "{} {}".format(output, " ".join(make_tuple(feat_idx_list, 0)))
    print( output, file=item_svm )

