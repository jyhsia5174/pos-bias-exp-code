#! /usr/bin/python3

# Filter by click number
#click_number=1000
click_number=3000

PATH = 'ad_filter_{}.csv'.format(click_number)
import csv

# Field 1, 2
keys1 = ['source_id', 'publisher_id', 'document_id']
keys2 = ['campaign_id', 'advertiser_id']

feat_dict = {}
incr_num1 = -1
incr_num2 = -1


def add_feat(key, value, field):
    global incr_num1
    global incr_num2
    global feat_dict
    real_key = "{0}:{1}".format(key, value)
    if real_key in feat_dict:
        return feat_dict[real_key]
    if field == 0:
        incr_num1 += 1
        feat_idx = incr_num1
    elif field == 1:
        incr_num2 += 1
        feat_idx = incr_num2
    feat_dict[real_key] = feat_idx
    return feat_idx


def make_tuple(feat_list,field):
    feat_str = ["%d:1" % i for i in feat_list]
    fnc = lambda x: "{}:{}".format(int(field), x)
    return list(map(fnc, feat_str))

item_svm = open('item.ffm', 'w')
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
        feat_idx_list.append(add_feat(key, line[key].strip(), 1))
    output = "{} {}".format(output, " ".join(make_tuple(feat_idx_list, 1)))
    print( output, file=item_svm )

