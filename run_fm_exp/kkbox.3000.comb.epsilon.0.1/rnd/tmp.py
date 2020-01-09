from sklearn.metrics import roc_auc_score, log_loss
import os, sys
import numpy as np
import torch  
import tqdm

root='./'
device='cpu'
Qs = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Qva')])
Ps = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Pva')])
Qs = torch.tensor(np.vstack([np.expand_dims(np.load(i).T, axis=0) for i in Qs])).to(device)
Ps = torch.tensor(np.vstack([np.expand_dims(np.load(i), axis=0) for i in Ps])).to(device)
item_num = Qs.size()[-1]
total_user_num=Ps.size()[1]
ctr_sum = np.zeros(item_num)

def dcg(target, y):
    tmp_y = sorted(y)
    return sum([(2**tmp_y[i] - 1)/np.log2(i+2) for i in range(len(y))])

with torch.no_grad(), open(sys.argv[1], 'r') as f:
    ys = list()
    targets = list()
    ndcg = 0
    #auc = 0
    #ll = 0
    count=0
    for i in tqdm.tqdm(range(0, Ps.size()[1]), smoothing=0, mininterval=1.0):
        items = list()
        line = f.readline().strip()
        items.extend([int(k.split(':')[0]) for k in line.split(' ', 1)[0].split(',')])
        target = [int(k.split(':')[1]) for k in line.split(' ', 1)[0].split(',')]
        targets.extend([int(k.split(':')[1]) for k in line.split(' ', 1)[0].split(',')])
        y = torch.sigmoid(torch.sum(torch.matmul(Ps[:, i, :], Qs[:, :, items]), 0))
        ys.extend(y.flatten().tolist())
        
        if sum(target) > 0:
            count += 1
            ndcg += dcg(target, y.flatten().tolist())
        #try:
        #    auc+=roc_auc_score(target, y.flatten().tolist())
        #except:
        #    auc+=0
        #try:    
        #    ll+=log_loss(target, y.flatten().tolist())
        #except:
        #    ll+=0

    #print(ll/Ps.size()[1], auc/Ps.size()[1])
    print(roc_auc_score(targets, ys), log_loss(targets, ys))
    print(ndcg/count)
