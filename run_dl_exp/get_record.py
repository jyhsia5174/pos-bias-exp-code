import os, sys

root = sys.argv[1]
mode = sys.argv[2]

f_path = os.path.join(root, "%s.record"%mode)
va_num = [2] 
#te_num = list(range(10, 30, 8)) 
logloss = list()
auc = list()
with open(f_path, 'r') as f:
    for i, line in enumerate(f):
        if i+1 in va_num:
            logloss.append(line.strip().split(' ')[-2])
            auc.append(line.strip().split(' ')[-1])
        if line.startswith(('ffm', 'dcn', 'ext', 'bi')):
            logloss.append(line.strip().split(' ')[-2])
            auc.append(line.strip().split(' ')[-1])

for i in logloss:
    print('%.4f'%float(i.strip()))
for i in auc:
    print('%.3f'%float(i.strip()))
