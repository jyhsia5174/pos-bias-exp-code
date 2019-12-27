import numpy as np
import os, sys
import torch  
import tqdm
from utility import recommend
from sklearn.metrics import roc_auc_score, log_loss

root = sys.argv[1]
gt_path = sys.argv[2]
device = 'cpu'
batch_size_of_user = 500
num_of_pos = 10
res = np.empty(batch_size_of_user*num_of_pos, dtype=np.int32)

Qs = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Qva')])
Ps = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Pva')]) 
#print(Qs, Ps)
Qs = torch.tensor(np.vstack([np.expand_dims(np.load(i).T, axis=0) for i in Qs])).to(device)  # (n_fields,embed_dim,item_num) 
Ps = torch.tensor(np.vstack([np.expand_dims(np.load(i), axis=0) for i in Ps])).to(device)  # (n_fields,context_num,embed_dim)
item_num = Qs.size()[-1]
embed_dim = Qs.size()[-2]
print(item_num, embed_dim)
#print(Qs.size(), Ps.size())

label_idxes, flags = list(), list()
with open(gt_path, 'r') as gt:
    pbar = tqdm.tqdm(gt, smoothing=0, mininterval=1.0)
    pbar.set_description('Loading gt:')
    for line in pbar:
        label, _ = line.strip().split(' ', 1)
        label = [tuple([int(i) for i in l.split(':')]) for l in label.split(',')]
        label_idx, flag = zip(*label)
        label_idxes.extend(list(label_idx))  # context_num*k
        flags.extend(list(flag)) # context_num*k

predicts = list()
with torch.no_grad():
    for i in tqdm.tqdm(range(0, Ps.size()[1], batch_size_of_user), smoothing=0, mininterval=1.0):
        num_of_user = Ps[:, i:i+batch_size_of_user, :].size()[1]
        y = torch.sigmoid(torch.sum(torch.matmul(Ps[:, i:i+num_of_user, :].view(-1, 1, embed_dim), Qs[:, :, label_idxes[i*num_of_pos:(i+num_of_user)*num_of_pos]].view(-1, embed_dim, num_of_user, num_of_pos).transpose(1, 2).view(-1, embed_dim, num_of_pos)).view(-1, num_of_user, num_of_pos), 0))  # (batch_size, num_of_pos)
        predicts.extend(torch.flatten(y).tolist())

print(roc_auc_score(flags, predicts), log_loss(flags, predicts))
