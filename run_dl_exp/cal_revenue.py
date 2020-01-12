import numpy as np
import os, sys
import pickle
from collections import defaultdict as ddict
np.random.seed(0)
pos = float(sys.argv[3])

stats = ddict(int)
with open(sys.argv[1], 'r') as preds, open(sys.argv[2], 'r') as gts:
    s = 0.
    for pline in preds:
        gline = gts.readline().strip()
        pline = pline.strip()
        gt = gline.split(' ', 1)[0]
        gt = gt.split(':')[0]
        preds = [tuple(p.split(':')) for p in pline.split(' ')]
        preds = list(zip(*preds))
        for i, ad in enumerate(preds[0]):
            if gt == ad:
                if np.random.rand() < pos**(i+1):
                    s += float(preds[-1][i])
                    stats[gt] += float(preds[-1][i])
                break
                #s += float(preds[1][i])*(0.9**(i+1))

print(s)
#print(sorted(stats.items(), key=lambda x:x[-1], reverse=True))
#pickle.dump(stats, open('revenue.stats', 'wb'))
