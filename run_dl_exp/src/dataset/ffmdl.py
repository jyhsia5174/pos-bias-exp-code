import os
import time
import sys
import torch
import numpy as np
import lmdb
import shutil
import struct
import subprocess
import random
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

class FFMDataset(Dataset):
    def __init__(self, dataset_path=None, data_prefix='tr', rebuild_cache=False, tr_max_dim=-1, read_flag=0):
        '''
        test_flag: 
            0: cntx_num*position_num
            1: cntx_num*item_num
            2: cntx_num*item_num, then randomly choose position_num items
            3: cntx_num*(item_num-position_num)
            4: cntx_num*(item_num-position_num), then randomly choose position_num items
        '''
        #self.tr_max_dim = tr_max_dim
        self.read_flag = read_flag
        data_path = os.path.join(dataset_path, data_prefix + '.ffm')
        item_path = os.path.join(dataset_path, 'item.ffm')
        assert Path(data_path).exists(), "%s does not exist!"%data_path
        cache_path = os.path.join(dataset_path, data_prefix + '.lmdb')

        # build cache
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(data_path, item_path, cache_path)

        # read data
        print('Reading data from %s.'%(cache_path))
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.long)
            self.item_num = np.frombuffer(txn.get(b'item_num'), dtype=np.int32)[0]
            self.pos_num = np.frombuffer(txn.get(b'pos_num'), dtype=np.int32)[0]
            self.max_ctx_num = np.frombuffer(txn.get(b'max_ctx_num'), dtype=np.int32)[0]
            self.max_item_num = np.frombuffer(txn.get(b'max_item_num'), dtype=np.int32)[0]
            self.items = np.frombuffer(txn.get(b'items'), dtype=np.float32).reshape((self.item_num, 3, -1))
            #self.length = self.pos_num*(txn.stat()['entries'] - 6)//2 if self.read_flag == 1 else self.item_num*(txn.stat()['entries'] - 6)//2
            self.length = (txn.stat()['entries'] - 6)//2 
            self.item_set = np.arange(self.item_num, dtype=np.int32)
            print('Totally %d items, %d dims, %d positions, %d samples'%(self.item_num, self.field_dims.sum(), self.pos_num, self.length))
    
    def __full_pass(self, data_path, item_path):
        ctx_col = 0
        item_col = 0
        item_num = 0
        feat_cnts = defaultdict(int)
        feat_cnts[0] = 1  # 0th field is reserved, it has only one dim
        print('Parsing files:')
        with open(item_path) as item, open(data_path) as data:
            pbar = tqdm(item, mininterval=1, smoothing=0.1)
            pbar.set_description('item_file')
            for line in pbar:
                item_num += 1
                line = line.strip().split()
                item_col = max(item_col, len(line))
                for feat in line:
                    field_id, feat_id, feat_val = feat.split(':')
                    field_id, feat_id, feat_val = int(field_id) + 1, int(feat_id), float(feat_val)
                    if field_id in feat_cnts:
                        feat_cnts[field_id] = max(feat_id, feat_cnts[field_id]) + 1
                    else:
                        feat_cnts[field_id] = feat_id + 1
            item_field_num = len(feat_cnts)

            pbar = tqdm(data, mininterval=1, smoothing=0.1)
            pbar.set_description('data_file')
            for line in pbar:
                line = line.strip().split()
                ctx_col = max(ctx_col, len(line) - 1)
                for feat in line[1:]:
                    field_id, feat_id, feat_val = feat.split(':')
                    field_id, feat_id, feat_val = item_field_num + int(field_id), int(feat_id), float(feat_val)
                    if field_id in feat_cnts:
                        feat_cnts[field_id] = max(feat_id, feat_cnts[field_id]) + 1
                    else:
                        feat_cnts[field_id] = feat_id + 1

            field_dims = np.zeros(len(feat_cnts), dtype=np.long)
            for k, val in feat_cnts.items():
                field_dims[k] = feat_cnts[k]
            
        return ctx_col + 1, item_col + 1, item_num, item_field_num, field_dims

    def __build_cache(self, data_path, item_path, cache_path):
        item_num = np.zeros(1, dtype=np.int32)
        pos_num = np.zeros(1, dtype=np.int32)
        max_ctx_num = np.zeros(1, dtype=np.int32)
        max_item_num = np.zeros(1, dtype=np.int32)

        max_ctx_num[0], max_item_num[0], item_num[0], item_field_num, field_dims = self.__full_pass(data_path, item_path)
        self.max_ctx_num = max_ctx_num[0]
        self.max_item_num = max_item_num[0]
        print('max_ctx_num:%d, max_item_num:%d'%(self.max_ctx_num, self.max_item_num))

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            i = 0
            items = np.zeros((item_num[0], 3, self.max_item_num), dtype=np.float32)
            with open(item_path, 'r') as fi:
                pbar = tqdm(fi, mininterval=1, smoothing=0.1)
                pbar.set_description('Create position dataset cache: setup lmdb for item')
                for line in pbar:
                    line = line.strip()
                    for _num, j in enumerate(line.split(' ')):
                        items[i, 0, _num] = int(j.split(':')[0]) + 1  # field_id
                        items[i, 1, _num] = int(j.split(':')[1])  # feat_id
                        items[i, 2, _num] = float(j.split(':')[2]) # val
                    i += 1
            assert item_num[0] == i
                
            for buf in self.__yield_buffer(data_path, item_field_num):
                with env.begin(write=True) as txn:
                    for item_key, item_array, ctx_key, ctx_array, pos_num[0] in buf:
                        txn.put(item_key, item_array)
                        txn.put(ctx_key, ctx_array)
        
            with env.begin(write=True) as txn:
                txn.put(b'items', items.tobytes())
                txn.put(b'field_dims', field_dims.tobytes())
                txn.put(b'item_num', item_num.tobytes())
                txn.put(b'pos_num', pos_num.tobytes())
                txn.put(b'max_ctx_num', max_ctx_num.tobytes())
                txn.put(b'max_item_num', max_item_num.tobytes())

    def __yield_buffer(self, data_path, item_field_num, buffer_size=int(1e6)):
        sample_idx, pos_num = 0, 0
        buf = list()
        with open(data_path, 'r') as fd:
            pbar = tqdm(fd, mininterval=1, smoothing=0.1)
            pbar.set_description('Create position dataset cache: setup lmdb for context')
            for line in pbar:
                line = line.strip()
                labels, context = line.split(' ', 1)
                labels = labels.strip().split(',')
                pos_num = len(labels)
                item_idx, item_value = zip(*[[int(j) for j in i.split(':')[:2]] for i in labels])
                ctx_field, ctx_idx, ctx_value = zip(*[[float(j) for j in i.split(':')] for i in context.split(' ')])
                item_array = np.zeros((2, pos_num), dtype=np.float32)
                item_array[0, :] = item_idx
                item_array[1, :] = item_value
                ctx_array = np.zeros((3, self.max_ctx_num), dtype=np.float32)
                ctx_field = item_field_num + np.array(ctx_field)
                ctx_array[0, :len(ctx_field)] = ctx_field
                ctx_array[1, :len(ctx_idx)] = ctx_idx
                ctx_array[2, :len(ctx_value)] = ctx_value

                buf.append((b'citem_%d'%sample_idx, item_array.tobytes(), b'ctx_%d'%sample_idx ,ctx_array.tobytes(), pos_num))
                sample_idx += 1
                if sample_idx % buffer_size == 0:
                    yield buf
                    buf.clear()
            yield buf

    def __len__(self):
        return self.length

    #@profile
    def __getitem__(self, idx):  # idx = 10*context_idx + pos
        if self.read_flag == 0:
            #context_idx, pos = divmod(idx, self.pos_num)
            with self.env.begin(write=False) as txn:
                item_array = np.frombuffer(txn.get(b'citem_%d'%idx), dtype=np.float32)
                ctx_array = np.frombuffer(txn.get(b'ctx_%d'%idx), dtype=np.float32).reshape(3, -1)

                item_idxes = item_array[:self.pos_num].astype(np.int32)
                flags = item_array[self.pos_num:]
                items = self.items[item_idxes, :, :]
                #ctx_idx = ctx_array[:self.max_ctx_num].astype(np.long)  # context
                #ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
            pos = np.arange(1, self.pos_num+1, dtype=np.long)
        elif self.read_flag == 1:
            #context_idx, item_idx = divmod(idx, self.item_num)
            with self.env.begin(write=False) as txn:
                item_array = np.frombuffer(txn.get(b'citem_%d'%idx), dtype=np.float32)
                ctx_array = np.frombuffer(txn.get(b'ctx_%d'%idx), dtype=np.float32).reshape(3, -1)

                items = self.items[:self.item_num, :, :]
                #ctx_idx = ctx_array[:self.max_ctx_num].astype(np.long)  # context
                #ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
            item_idxes = np.arange(self.item_num, dtype=np.int32)
            flags = np.ones(self.item_num)*-1
            pos = np.zeros(self.item_num)
        elif self.read_flag == 2:
            with self.env.begin(write=False) as txn:
                item_array = np.frombuffer(txn.get(b'citem_%d'%idx), dtype=np.float32)
                ctx_array = np.frombuffer(txn.get(b'ctx_%d'%idx), dtype=np.float32).reshape(3, -1)
                item_idxes = np.setxor1d(item_array[:self.pos_num].astype(np.int32), self.item_set, True)
                item_idxes = np.hstack((item_array[:self.pos_num].astype(np.int32), np.random.choice(item_idxes, self.pos_num, replace=False)))

                items = self.items[item_idxes, :, :]
                flags = np.hstack((item_array[self.pos_num:], np.zeros(self.pos_num)))
            pos = np.hstack((np.arange(1, self.pos_num+1, dtype=np.long), np.zeros(self.pos_num)))
        #elif self.read_flag == 2:
        #    #context_idx, item_idx = divmod(idx, self.item_num)
        #    with self.env.begin(write=False) as txn:
        #        item_array = np.frombuffer(txn.get(b'citem_%d'%idx), dtype=np.float32)
        #        ctx_array = np.frombuffer(txn.get(b'ctx_%d'%idx), dtype=np.float32).reshape(3, -1)

        #        item_idxes = np.random.choice(self.item_set, self.pos_num, replace=False)
        #        items = self.items[item_idxes, :, :]
        #        #ctx_idx = ctx_array[:self.max_ctx_num].astype(np.long)  # context
        #        #ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
        #    flags = np.ones(self.pos_num)*-1
        #    pos = np.zeros(self.pos_num)
        #elif self.read_flag == 3:
        #    with self.env.begin(write=False) as txn:
        #        item_array = np.frombuffer(txn.get(b'citem_%d'%idx), dtype=np.float32)
        #        ctx_array = np.frombuffer(txn.get(b'ctx_%d'%idx), dtype=np.float32).reshape(3, -1)
        #        item_idxes = np.setxor1d(item_array[:self.pos_num].astype(np.int32), self.item_set, True)

        #        items = self.items[item_idxes, :, :]
        #        #ctx_idx = ctx_array[:self.max_ctx_num].astype(np.long)  # context
        #        #ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
        #    flags = np.ones(self.item_num - self.pos_num)*-1
        #    pos = np.zeros(self.item_num - self.pos_num)
        #elif self.read_flag == 4:
        #    with self.env.begin(write=False) as txn:
        #        item_array = np.frombuffer(txn.get(b'citem_%d'%idx), dtype=np.float32)
        #        ctx_array = np.frombuffer(txn.get(b'ctx_%d'%idx), dtype=np.float32).reshape(3, -1)
        #        item_idxes = np.setxor1d(item_array[:self.pos_num].astype(np.int32), self.item_set, True)
        #        item_idxes = np.random.choice(item_idxes, self.pos_num, replace=False)

        #        items = self.items[item_idxes, :, :]
        #        #ctx_idx = ctx_array[:self.max_ctx_num].astype(np.long)  # context
        #        #ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
        #    flags = np.ones(self.pos_num)*-1
        #    pos = np.zeros(self.pos_num)
        else:
            raise ValueError('Wrong flag for reading data'%self.read_flag)

        if self.read_flag == 0:
            return np.tile(ctx_array, (self.pos_num, 1, 1)), items, flags, pos, item_idxes#, np.tile(ctx_value, (self.pos_num, 1))  # pos \in {1,2,...9,10}, 0 for no-position
        elif self.read_flag == 1:
            return np.tile(ctx_array, (self.item_num, 1, 1)), items, flags, pos, item_idxes#, np.tile(ctx_value, (self.item_num, 1))  # pos \in {1,2,...9,10}, 0 for no-position
        elif self.read_flag == 2:
            return np.tile(ctx_array, (self.pos_num*2, 1, 1)), items, flags, pos, item_idxes#, np.tile(ctx_value, (self.pos_num, 1))  
        #elif self.read_flag == 2:
        #    return np.tile(ctx_idx, (self.pos_num, 1)), items, flags, pos, item_idxes, np.tile(ctx_value, (self.pos_num, 1))  
        #elif self.read_flag == 3:
        #    return np.tile(ctx_idx, (self.item_num - self.pos_num, 1)), items, flags, pos, item_idxes, np.tile(ctx_value, (self.item_num - self.pos_num, 1)) 
        #else:
        #    return np.tile(ctx_idx, (self.pos_num, 1)), items, flags, pos, item_idxes, np.tile(ctx_value, (self.pos_num, 1))  


    def get_max_dim(self):
        return self.field_dims.sum() 

    def get_item_num(self):
        return self.item_num

if __name__ == '__main__':
    #@profile
    #def collate_fn_for_dssm(batch):
    #    print(batch)
    #    context = [torch.LongTensor(i['context']) for i in batch]
    #    value = [torch.FloatTensor(i['value']) for i in batch]
    #    item = [torch.LongTensor(i['item']) for i in batch]
    #    label = [i['label'] for i in batch]
    #    pos = [i['pos'] for i in batch]
    #    item = rnn_utils.pad_sequence(item, batch_first=True, padding_value=0)
    #    context = rnn_utils.pad_sequence(context, batch_first=True, padding_value=0)
    #    value = rnn_utils.pad_sequence(value, batch_first=True, padding_value=0)
    #    return context, item, torch.FloatTensor(label), torch.FloatTensor(pos).unsqueeze(-1), value
    def set_seed(x=0):
        np.random.seed(x)
        random.seed(x)
        torch.manual_seed(x)
        torch.cuda.manual_seed_all(x)
        torch.backends.cudnn.deterministic = True
        return

    def worker_init_fn(x, inseed=0):
        seed = inseed + x
        set_seed(seed)
        return

    class SimDataset(Dataset):
        def __init__(self, dataset1, dataset2):
            assert len(dataset1) == len(dataset2), "Can't combine 2 datasets for their different length!"
            self.dataset1 = dataset1 # datasets should be sorted!
            self.dataset2 = dataset2

        def __getitem__(self, index):
            x1 = self.dataset1[index]
            x2 = self.dataset2[index]

            return x1, x2

        def __len__(self):
            return len(self.dataset1)

    #@profile
    def main(dataset):#, imp_dataset):
        from torch.utils.data import SubsetRandomSampler as srs
        device='cuda:0'
        data_loader = DataLoader(dataset, batch_size=50, num_workers=0, shuffle=False)
        #sim_dataset = SimDataset(dataset, imp_dataset)
        #data_loader = DataLoader(sim_dataset, batch_size=10, num_workers=8, shuffle=True) #worker_init_fn=worker_init_fn, sampler=srs(sample_list))
        print(dataset.field_dims)
        pbar = tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
        for i, data_pack in enumerate(pbar):
        #for i, (data_pack, imp_data_pack) in enumerate(pbar):
        #for i, (data_pack, imp_data_pack) in enumerate(zip(data_loader, imp_data_loader)):
            context, item, target, pos, item_idxes = data_pack
            #imp_context, imp_item, imp_target, imp_pos, imp_item_idxes, imp_value = imp_data_pack
            #context, item, target, pos, value = \
            #        context.view(tuple(-1 if i==0 else _s for i, _s in enumerate(context.size()[1:]))).to(device, torch.long), \
            #        item.view(tuple(-1 if i==0 else _s for i, _s in enumerate(item.size()[1:]))).to(device, torch.long), \
            #        target.view(tuple(-1 if i==0 else _s for i, _s in enumerate(target.size()[1:]))).to(device, torch.float), \
            #        pos.view(tuple(-1 if i==0 else _s for i, _s in enumerate(pos.size()[1:]))).to(device, torch.long), \
            #        value.view(tuple(-1 if i==0 else _s for i, _s in enumerate(value.size()[1:]))).to(device, torch.float)
            print(context[30:31], item[30:31], target[30:31], pos[30:31], item_idxes[30:31])#, value[:])
            print(context.size(), item.size(), target.size(), pos.size())#, value.size())
            break
            #print(item_idxes)
            #print(context.size(), imp_context.size())
            #if (context[:, 0, :] - imp_context[:, 0, :]).sum().numpy() < 1e-9:
            #    print('Same!')
            #else:
            #    print((context[:, 0, :], imp_context[:, 0, :]))
            #    break
            #print(context, item, target, pos, value)
            #if i > -1 + 100: 
            #    break
            #print(idx, data, label, pos)

    dataset = FFMDataset(dataset_path=sys.argv[1], data_prefix='va', rebuild_cache=False, tr_max_dim=-1, read_flag=2)
    #imp_dataset = FFMDataset(dataset_path=sys.argv[1], data_prefix='va', rebuild_cache=False, tr_max_dim=-1, read_flag=2)
    main(dataset)#, imp_dataset)
