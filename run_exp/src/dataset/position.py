import os
import time
import sys
import torch
import numpy as np
import lmdb
import shutil
import struct
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class PositionDataset(Dataset):
    def __init__(self, dataset_path=None, data_prefix='tr', rebuild_cache=False, tr_max_dim=-1):
        #self.items = []
        #self.contexts = []
        #self.labels = []
        ##self.len = 0  # number of samples
        #if training:
        #    self.n_feature = 0  # dim of features
        #elif tr_n_feature > 0.:
        #    self.n_feature = tr_n_feature
        #else:
        #    raise ValueError("tr_n_feature is not set!")

        #files = os.listdir(root_dir)
        #data_file = 'va.svm' if training else 'va.svm'
        #assert data_file in files, "%s does not exist!"%data_file
        #assert 'item.svm' in files, "item.svm does not exist!"
        #data_file = os.path.join(root_dir, data_file)
        #item_file = os.path.join(root_dir, 'item.svm')
        #with open(data_file, 'r') as data, open(item_file, 'r') as item:
        #    for line in item.readlines():
        #        line = line.strip()
        #        items = [int(i.split(':')[0]) for i in line.split(' ')]
        #        self.items.append(items)

        #    for line in data.readlines():
        #        line = line.strip()
        #        l, c = line.split(' ', 1)
        #        l = l.split(',')
        #        if training:
        #            c = sorted([int(i.split(':')[0]) for i in c.split(' ')])
        #            n_feature = c[-1]
        #            if n_feature > self.n_feature:
        #                self.n_feature = n_feature
        #        else:
        #            c = sorted([int(i.split(':')[0]) for i in c.split(' ') if int(i.split(':')[0]) <= tr_n_feature - 11])
        #        self.labels.append(l)
        #        self.contexts.append(c)
        #if training:
        #    self.n_feature += 11  # add idx for pos and padding feature
        #print('max dim:%d'%self.n_feature)
        ##print(len(self.items), len(self.contexts), len(self.labels))
        
        self.tr_max_dim = tr_max_dim
        data_path = os.path.join(dataset_path, data_prefix + '.svm')
        item_path = os.path.join(dataset_path, 'item.svm')
        assert Path(data_path).exists(), "%s does not exist!"%data_path
        cache_path = os.path.join(dataset_path, data_prefix + '.lmdb')

        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(data_path, item_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.max_dim = np.frombuffer(txn.get(b'max_dim'), dtype=np.int32)[0] + 1  # idx from 0 to max_dim_in_svmfile, 0 for padding
    
    def __build_cache(self, data_path, item_path, cache_path):
        max_dim = np.zeros(1, dtype=np.int32)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            for buf in self.__yield_buffer(data_path, item_path):
                with env.begin(write=True) as txn:
                    for key, value, max_dim_buf in buf:
                        txn.put(key, value)
                        if  max_dim_buf > max_dim[0]:
                            max_dim[0] = max_dim_buf
            with env.begin(write=True) as txn:
                txn.put(b'max_dim', max_dim.tobytes())

    def __yield_buffer(self, data_path, item_path, buffer_size=int(1e5)):
        sample_idx, max_dim = 0, 0
        buf, items = list(), list()
        with open(data_path, 'r') as fd, open(item_path, 'r') as fi:
            for line in fi.readlines():
                line = line.strip()
                item = [int(i.split(':')[0]) for i in line.split(' ')]
                items.append(item)

            pbar = tqdm(fd, mininterval=1, smoothing=0.1)
            pbar.set_description('Create position dataset cache: setup lmdb')
            for line in pbar:
                line = line.strip()
                labels, context = line.split(' ', 1)
                labels = labels.split(',')
                context = [int(i.split(':')[0]) for i in context.split(' ')]
                for pos, l in enumerate(labels):
                    item_idx, flag = l.split(':')
                    item = items[int(item_idx)]
                    feature = sorted(item + context)
                    if  feature[-1] > max_dim:
                        max_dim = feature[-1]
                    feature = [int(flag), item_idx, pos] + feature
                    feature = np.array(feature, dtype=np.int32)  # [label, item_idx, position, feature_idx]
                    buf.append((struct.pack('>I', sample_idx), feature.tobytes(), max_dim))
                    sample_idx += 1
                    if sample_idx % buffer_size == 0:
                        yield buf
                        buf.clear()
            yield buf

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(txn.get(struct.pack('>I', idx)), dtype=np.int32)
            data = np_array[3:]
        if self.tr_max_dim > 0:
            data = data[data <= self.tr_max_dim]
        return {'data':data, 'label':np_array[0], 'pos':np_array[2], 'item_idx':np_array[1]}

    def get_max_dim(self):
        return self.max_dim


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = PositionDataset(dataset_path='../../../data/random', data_prefix='va', rebuild_cache=False, tr_max_dim=1000000)
    print('Start loading!')
    #f = open('pos.svm', 'w')
    print(len(dataset))
    print(dataset.get_max_dim())
    for i in range(len(dataset)):
        sample_batch = dataset.__getitem__(i)
        #if 1 in sample_batched['label']:
        #    print(i_batch, sample_batched)
        #    break
        print(i, sample_batch)
        if i> 10:
            break
        #data = ['%d:1'%i for i in sorted(sample_batched['data'])] 
        #label = "+1" if sample_batched['label'] else "-1"
        #f.write("%s %s\n"%(label, " ".join(data)))
    #f.close()
