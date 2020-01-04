import numpy as np
import os, sys
from collections import defaultdict as ddict
np.random.seed(0)

stats = ddict(int)
with open(sys.argv[1], 'r') as preds, open(sys.argv[2], 'r') as gts:
    s = 0.
    for pline in preds:
        gline = gts.readline()
        pline = pline.strip()
        gt = gline.split(' ', 1)[0]
        gt = gt.split(':')[0]
        preds = [tuple(p.split(':')) for p in pline.split(' ')]
        preds = list(zip(*preds))
        for i, ad in enumerate(preds[0]):
            if gt == ad:
                if np.random.rand() < 0.5**(i+1):
                    s += float(preds[2][i])
                    stats[gt] += float(preds[2][i])
                #s += float(preds[1][i])*(0.9**(i+1))

print(s)
#print(sorted(stats.items(), key=lambda x:x[-1], reverse=True))
