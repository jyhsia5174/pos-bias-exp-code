import os
import sys
import numpy as np
from collections import defaultdict as ddict 
np.random.seed(0)

root = sys.argv[1]
rnd_path = os.path.join(root, 'random_trva.svm')  # rnd trva
det_path = os.path.join(root, 'det_tr.svm')  # det tr
mix_tr_path = os.path.join(root, 'select_tr.svm')  # mix tr
mix_trva_path = os.path.join(root, 'select_trva.svm')  # mix trva

obs_tr_path = 'obs_tr.svm'  # 90% rnd_tr+det_tr
obs_va_path = 'obs_va.svm'  # 10% rnd_tr+det_tr
obs_trva_path = 'obs_trva.svm'  # rnd_tr+det_tr
obs_full_path = 'obs_full.svm'  # rnd_tr+det_tr+rnd_va

def main():
    rnd_dict = ddict(str)
    det_dict = ddict(str)
    rnd_va_dict = ddict(str)
    with open(rnd_path, 'r') as rnd:
        for line in rnd:
            line = line.strip().split(' ', 1)
            rnd_dict[line[1]] = line[0]
    
    with open(det_path, 'r') as det:
        for line in det:
            line = line.strip().split(' ', 1)
            det_dict[line[1]] = line[0]
    
    with open(mix_tr_path, 'r') as mix_tr, open(obs_tr_path, 'w') as tr, open(obs_va_path, 'w') as va, \
            open(obs_trva_path, 'w') as trva:  #, open(obs_full_path, 'w') as full:
        for line in mix_tr:
            line = line.strip().split(' ', 1)
            trl = []
            val = []

            for label in line[0].split(','):
                if line[1] in rnd_dict:  # maybe have random sample
                    if label in rnd_dict[line[1]]:  # random sample
                        label = label.replace(':1', ':11')   # 11, click-label, bias-label
                        label = label.replace(':0', ':1')   # 01
                    else:  # det sample
                        label = label.replace(':1', ':10')  # 10
                        #label = label.replace(':0', ':0')  # 00
                else:  # only have det sample
                    assert line[1] in det_dict and label in det_dict[line[1]], "%s %s"%(line[0], line[1])
                    label = label.replace(':1', ':10')  # 10 
                    #label = label.replace(':0', ':0')  # 00

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

    with open(mix_trva_path, 'r') as mix_trva, open(obs_full_path, 'w') as full:
        for line in mix_trva:
            line = line.strip().split(' ', 1)
            ls = []

            for label in line[0].split(','):
                if line[1] in rnd_dict:  # maybe have random samples
                    if label in rnd_dict[line[1]]:  # random sample
                        label = label.replace(':1', ':11')   # 11, click-label, bias-label
                        label = label.replace(':0', ':1')   # 01
                    else:  # det sample
                        label = label.replace(':1', ':10')  # 10
                        #label = label.replace(':0', ':0')  # 00
                else:  # only have det samples
                    assert line[1] in det_dict and label in det_dict[line[1]], "%s %s"%(line[0], line[1])
                    label = label.replace(':1', ':10')  # 10 
                    #label = label.replace(':0', ':0')  # 00
                ls.append(label)

            full.write('%s %s\n'%(','.join(ls), line[1]))


main()


