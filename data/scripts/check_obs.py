import os, sys

root = sys.argv[1]

det_tr = os.path.join(root, 'det_tr.svm')
rnd_tr = os.path.join(root, 'random_tr.svm')
rnd_va = os.path.join(root, 'random_va.svm')
obs_tr = 'obs_tr.svm'
obs_va = 'obs_va.svm'
obs_trva = 'obs_trva.svm'
obs_full = 'obs_full.svm'

def count_line(f):
    c = 0
    for line in f:
        line = line.strip().split(' ', 1)
        c+=len(line[0].split(','))
    return c

for i in [det_tr, rnd_tr, obs_tr, obs_va, obs_trva, obs_full, rnd_va]:
    print(count_line(open(i)))
