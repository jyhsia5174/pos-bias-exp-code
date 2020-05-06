import os
import sys
import numpy as np
from collections import defaultdict as ddict 
np.random.seed(0)

root = sys.argv[1]
det_path = os.path.join(root, 'det_trva.svm')
rnd_path = os.path.join(root, 'random_trva.svm')
mix_path = os.path.join(root, 'select_trva.svm')

obs_tr_path = 'obs_tr.svm'
obs_va_path = 'obs_va.svm'
obs_trva_path = 'obs_trva.svm'

def main():
    rnd_dict = ddict(str)
    det_dict = ddict(str)
    with open(rnd_path, 'r') as rnd:
        for line in rnd:
            line = line.strip().split(' ', 1)
            rnd_dict[line[1]] = line[0]
    
    with open(det_path, 'r') as det:
        for line in det:
            line = line.strip().split(' ', 1)
            det_dict[line[1]] = line[0]
    
    with open(mix_path, 'r') as mix, open(obs_tr_path, 'w') as tr, \
            open(obs_va_path, 'w') as va, open(obs_trva_path, 'w') as trva:
        for line in mix:
            line = line.strip().split(' ', 1)
            trl = []
            val = []

            if line[1] in det_dict:  # mix
                for label in line[0].split(','):
                    if label in det_dict[line[1]]:
                        label = label.replace(':1', ':0') 
                    else:
                        label = label.replace(':0', ':1') 
                    r = np.random.rand()
                    if r > 0.1:
                        trl.append(label)
                    else:
                        val.append(label)
            else:  # random
                for label in line[0].split(','):
                    label = label.replace(':0', ':1') 
                    r = np.random.rand()
                    if r > 0.1:
                        trl.append(label)
                    else:
                        val.append(label)

            if len(trl):
                tr.write('%s %s\n'%(','.join(trl), line[1]))
            if len(val):
                va.write('%s %s\n'%(','.join(val), line[1]))
            trva.write('%s %s\n'%(','.join(trl+val), line[1]))


main()


