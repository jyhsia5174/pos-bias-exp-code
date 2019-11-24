import os
import time
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PositionDataset(Dataset):
    def __init__(self, root_dir, tr_n_feature=0, training=True, transform=None):
        self.items = []
        self.contexts = []
        self.labels = []
        #self.len = 0  # number of samples
        if training:
            self.n_feature = 0  # dim of features
        elif tr_n_feature > 0.:
            self.n_feature = tr_n_feature
        else:
            raise ValueError("tr_n_feature is not set!")

        files = os.listdir(root_dir)
        data_file = 'va.svm' if training else 'va.svm'
        assert data_file in files, "%s does not exist!"%data_file
        assert 'item.svm' in files, "item.svm does not exist!"
        data_file = os.path.join(root_dir, data_file)
        item_file = os.path.join(root_dir, 'item.svm')
        with open(data_file, 'r') as data, open(item_file, 'r') as item:
            for line in item.readlines():
                line = line.strip()
                items = [int(i.split(':')[0]) for i in line.split(' ')]
                self.items.append(items)

            for line in data.readlines():
                line = line.strip()
                l, c = line.split(' ', 1)
                l = l.split(',')
                if training:
                    c = sorted([int(i.split(':')[0]) for i in c.split(' ')])
                    n_feature = c[-1]
                    if n_feature > self.n_feature:
                        self.n_feature = n_feature
                else:
                    c = sorted([int(i.split(':')[0]) for i in c.split(' ') if int(i.split(':')[0]) <= tr_n_feature - 11])
                self.labels.append(l)
                self.contexts.append(c)
        if training:
            self.n_feature += 11  # add idx for pos and padding feature
        print('max dim:%d'%self.n_feature)
        #print(len(self.items), len(self.contexts), len(self.labels))
    
    def __len__(self):
        return 10*len(self.contexts) # 10 ads per context

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()[0]
        
        context_idx = int(idx/10)
        label_idx = int(idx%10)
        context = self.contexts[context_idx]
        label = self.labels[context_idx][label_idx]
        item_idx, flag = label.split(':')
        data = self.items[int(item_idx)] + context
        #print(idx, context_idx, label_idx, item_idx)

        return {'data':data, 'label':int(flag), 'pos':label_idx + self.n_feature}

    def get_n_feature(self):
        return self.n_feature


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = PositionDataset('../../data/random', training=False)
    #data = dataset.__getitem__(102*10+7)
    #print(sum(data['data']), data['label'])
    #dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    print('Start loading!')
    f = open('pos.svm', 'w')
    for i in range(len(dataset)):
        sample_batched = dataset.__getitem__(i)
        #if 1 in sample_batched['label']:
        #    print(i_batch, sample_batched)
        #    break
        data = ['%d:1'%i for i in sorted(sample_batched['data'])] 
        label = "+1" if sample_batched['label'] else "-1"
        f.write("%s %s\n"%(label, " ".join(data)))
    f.close()
