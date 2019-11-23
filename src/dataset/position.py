import os
import time
import sys
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class PositionDataset(Dataset):
    def __init__(self, root_dir, training=True, transform=None):
        self.items = []
        self.contexts = []
        self.labels = []
        self.len = 0  # number of samples
        self.n_feature = 0  # dim of features

        files = os.listdir(root_dir)
        data_file = 'tr.svm' if training else 'va.svm'
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
                c = sorted([int(i.split(':')[0]) for i in c.split(' ')])
                self.labels.append(l)
                self.contexts.append(c)
                n_feature = c[-1]
                if n_feature > self.n_feature:
                    self.n_feature = n_feature
        self.n_feature += 100000    # drop feature not in trainset  
        print('max dim:%d'%self.n_feature)
    
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

        return {'data':data, 'label':int(flag), 'pos':label_idx + self.n_feature - 10}


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = PositionDataset('../../data/random', training=False)
    #data = dataset.__getitem__(102*10+7)
    #print(sum(data['data']), data['label'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    print('Start loading!')
    for i_batch, sample_batched in enumerate(dataloader):
        if 1 in sample_batched['label']:
            print(i_batch, sample_batched)
            break
