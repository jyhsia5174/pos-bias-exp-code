import numpy as np
import os, sys
import pickle
from collections import defaultdict as ddict
from sklearn.metrics import roc_auc_score, log_loss
np.random.seed(0)
try:
    bias_base = float(sys.argv[3])
except:
    bias_base = 0.9

stats = ddict(int)
probs = list()
probs2 = list()
truths = list()
const_truths = list()
pos_truths = list()

pos_biases = [bias_base**i for i in range(10)]
const_pos_bias = sum(pos_biases)/10.

with open(sys.argv[1], 'r') as preds, open(sys.argv[2], 'r') as gts:
    s = 0.
    for pline in preds:
        gline = gts.readline().strip()
        pline = pline.strip()
        gt = gline.split(' ', 1)[0]
        gt = gt.split(':')[0]
        preds = [tuple(p.split(':')) for p in pline.split(' ')]
        preds = list(zip(*preds))
        probs.extend([float(p) for p in preds[1]])

        idxes = np.arange(10)
        np.random.shuffle(idxes)
        for i, k in enumerate(zip(preds[0], idxes)):
            ad, idx = k
            ad2 = preds[0][idx]
            probs2.append(float(preds[1][idx]))
            rnd = np.random.rand()
            if gt == ad:
                truths.append(1)
                if rnd < pos_biases[i]:
                    s += float(preds[-1][i])
                    stats[gt] += float(preds[-1][i])
                if rnd <= const_pos_bias:
                    const_truths.append(1)
                else:
                    const_truths.append(0)
                #break
            else:
                truths.append(0)
                const_truths.append(0)
                #s += float(preds[1][i])*(0.9**(i+1))
            if gt == ad2:
                if rnd <= pos_biases[i]:
                    pos_truths.append(1)
                else:
                    pos_truths.append(0)
            else:
                pos_truths.append(0)

print(s)
#print(roc_auc_score(truths, probs), log_loss(truths, probs))
#print(roc_auc_score(const_truths, probs), log_loss(const_truths, probs))
#print(roc_auc_score(pos_truths, probs), log_loss(pos_truths, probs2))
#print(sorted(stats.items(), key=lambda x:x[-1], reverse=True))
#pickle.dump(stats, open('revenue.stats', 'wb'))
