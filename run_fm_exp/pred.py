import numpy as np
import os, sys
import torch  
import tqdm
from utility import recommend

root = sys.argv[1]
device = 'cpu'
batch_size_of_user = 5
num_of_pos=10
res = np.empty(batch_size_of_user*num_of_pos, dtype=np.int32)

Qs = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Qva')])
Ps = sorted([os.path.join(root, i) for i in os.listdir(root) if i.startswith('Pva')])
#print(Qs, Ps)
Qs = torch.tensor(np.vstack([np.expand_dims(np.load(i).T, axis=0) for i in Qs])).to(device)
Ps = torch.tensor(np.vstack([np.expand_dims(np.load(i), axis=0) for i in Ps])).to(device)
item_num = Qs.size()[-1]
#print(Qs.size(), Ps.size())

rngs = [np.random.RandomState(seed) for seed in [0,3,4,5,6]]
bids = np.empty((len(rngs), item_num)) 
for i, rng in enumerate(rngs):
    bids[i, :] = rng.gamma(20, 1/0.4, item_num)
bids = torch.tensor(bids).to(device)

with torch.no_grad():
    for i in tqdm.tqdm(range(0, Ps.size()[1], batch_size_of_user), smoothing=0, mininterval=1.0):
        y = torch.sigmoid(torch.sum(torch.matmul(Ps[:, i:i+batch_size_of_user, :], Qs), 0))
        num_of_user = y.size()[0]
        for j in range(len(rngs)):
            with open(os.path.join(root, 'tmp.pred.%d'%j), 'a') as fp:
                out = y.flatten()*(bids[j, :].repeat(num_of_user))
                recommend.get_top_k_by_greedy(out.cpu().numpy().flatten(), num_of_user, item_num, num_of_pos, res[:num_of_user*num_of_pos])
                _res = res[:num_of_user*num_of_pos].reshape(num_of_user, num_of_pos)
                for r in range(num_of_user):
                    tmp = ['%d:%.4f'%(ad, bids[j, ad]) for ad in _res[r, :]]
                    fp.write('%s\n'%(' '.join(tmp)))
 
