import os, sys
import os.path as osp
from tensorboardX import SummaryWriter 

root = sys.argv[1]
logs = [root]
#logs = [osp.join(root, f) for f in os.listdir(root) if f.endswith('log')]
for l in logs:
    writer = SummaryWriter('runs/%s'%(l.split('/')[-3]+'_'+osp.basename(l)[:-4]))
    with open(l) as l:
        for i, line in enumerate(l):
            line = line.strip().split('\t')
            for val in line:
                if 'epoch' in val:
                    pass
                else:
                    n, v = val.split(':')
                    writer.add_scalar(n, float(v), i)
    writer.close()

