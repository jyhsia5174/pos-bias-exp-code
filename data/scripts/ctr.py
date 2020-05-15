import os, sys
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss

item = 'item.svm'
rnd_tr = 'random_tr.svm'
rnd_trva = 'random_trva.svm'
det_trva = 'det_trva.svm'
te = 'rnd_gt.svm'

item_num = 0
for line in open(item):
    item_num += 1

stats = np.zeros((2, item_num))
for line in open(rnd_tr):
    line = line.strip().split(' ', 1)
    for l in line[0].split(','):
        idx, click = l.split(':')
        stats[0, int(idx)] += int(click)
        stats[1, int(idx)] += 1
np.save('rnd_stats_tr.npy', stats)

stats = np.zeros((2, item_num))
for line in open(rnd_trva):
    line = line.strip().split(' ', 1)
    for l in line[0].split(','):
        idx, click = l.split(':')
        stats[0, int(idx)] += int(click)
        stats[1, int(idx)] += 1
np.save('rnd_stats_trva.npy', stats)

det_stats = np.zeros((2, item_num))
for line in open(det_trva):
    line = line.strip().split(' ', 1)
    for l in line[0].split(','):
        idx, click = l.split(':')
        det_stats[0, int(idx)] += int(click)
        det_stats[1, int(idx)] += 1

label = []
for line in open(te):
    line = line.strip().split(' ', 1)
    for l in line[0].split(','):
        label.append(int(l.split(':')[-1]))

rnd_pctr = stats.sum(axis=1)[0]/stats.sum(axis=1)[1]
det_pctr = det_stats.sum(axis=1)[0]/det_stats.sum(axis=1)[1]
print("rnd_pctr_trva:%f, det_pctr_trva:%f"%(rnd_pctr, det_pctr))
print('avg_st auc-%.4f logloss-%.4f'%(roc_auc_score(label, np.ones_like(label)*rnd_pctr), log_loss(label, np.ones_like(label)*rnd_pctr)))
print('avg_sc auc-%.4f logloss-%.4f'%(roc_auc_score(label, np.ones_like(label)*det_pctr), log_loss(label, np.ones_like(label)*det_pctr)))
