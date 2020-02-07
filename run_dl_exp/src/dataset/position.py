import os
import time
import sys
import torch
import numpy as np
import lmdb
import shutil
import struct
import subprocess
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

class PositionDataset(Dataset):
    def __init__(self, dataset_path=None, data_prefix='tr', rebuild_cache=False, tr_max_dim=-1, test_flag=False):
        self.tr_max_dim = tr_max_dim
        self.test_flag = test_flag
        data_path = os.path.join(dataset_path, data_prefix + '.svm')
        item_path = os.path.join(dataset_path, 'item.svm')
        assert Path(data_path).exists(), "%s does not exist!"%data_path
        cache_path = os.path.join(dataset_path, data_prefix + '.lmdb')

        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(data_path, item_path, cache_path)
        print('Reading data from %s.'%(cache_path))
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.max_dim = np.frombuffer(txn.get(b'max_dim'), dtype=np.int32)[0] + 1  # idx from 0 to max_dim_in_svmfile, 0 for padding
            self.item_num = np.frombuffer(txn.get(b'item_num'), dtype=np.int32)[0]
            self.pos_num = np.frombuffer(txn.get(b'pos_num'), dtype=np.int32)[0]
            self.max_ctx_num = np.frombuffer(txn.get(b'max_ctx_num'), dtype=np.int32)[0]
            self.max_item_num = np.frombuffer(txn.get(b'max_item_num'), dtype=np.int32)[0]
            self.length = self.pos_num*(txn.stat()['entries'] - self.item_num - 5)//2 if not self.test_flag else self.item_num*(txn.stat()['entries'] - self.item_num - 5)//2
            print('Totally %d items, %d dims, %d positions, %d samples'%(self.item_num, self.max_dim, self.pos_num, self.length))
    
    def __build_cache(self, data_path, item_path, cache_path):
        max_dim = np.zeros(1, dtype=np.int32)
        item_num = np.zeros(1, dtype=np.int32)
        pos_num = np.zeros(1, dtype=np.int32)
        max_ctx_num = np.zeros(1, dtype=np.int32)
        max_item_num = np.zeros(1, dtype=np.int32)

        ctx_col = subprocess.run("awk 'BEGIN{max = 0}{if (NF+0 >= max+0) max=NF}END{print max}' %s"%data_path, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
        item_col = subprocess.run("awk 'BEGIN{max = 0}{if (NF+0 >= max+0) max=NF}END{print max}' %s"%item_path, shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8")
        if ctx_col.returncode or item_col.returncode:
            raise ValueError('Can get %s or %s max_col_num!'%(data_path, item_path))
        else:
            self.max_ctx_num = int(ctx_col.stdout.strip()) - 1
            self.max_item_num = int(item_col.stdout.strip())
            max_ctx_num[0] = self.max_ctx_num
            max_item_num[0] = self.max_item_num
            print('max_ctx_num:%d, max_item_num:%d'%(self.max_ctx_num, self.max_item_num))

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            i = 0
            with open(item_path, 'r') as fi:
                pbar = tqdm(fi, mininterval=1, smoothing=0.1)
                pbar.set_description('Create position dataset cache: setup lmdb for item')
                for line in pbar:
                    line = line.strip()
                    item = np.zeros(self.max_item_num, dtype=np.int32)
                    for _num, j in enumerate(line.split(' ')):
                        item[_num] = int(j.split(':')[0])
                    with env.begin(write=True) as txn:
                        txn.put(b'item_%d'%i, item.tobytes())
                        i += 1
            item_num[0] = i
                
            for buf in self.__yield_buffer(data_path):
                with env.begin(write=True) as txn:
                    for item_key, item_array, ctx_key, ctx_array, max_dim_buf, pos_num[0] in buf:
                        txn.put(item_key, item_array)
                        txn.put(ctx_key, ctx_array)
                        if  max_dim_buf > max_dim[0]:
                            max_dim[0] = max_dim_buf
        
            with env.begin(write=True) as txn:
                txn.put(b'max_dim', max_dim.tobytes())
                txn.put(b'item_num', item_num.tobytes())
                txn.put(b'pos_num', pos_num.tobytes())
                txn.put(b'max_ctx_num', max_ctx_num.tobytes())
                txn.put(b'max_item_num', max_item_num.tobytes())

    def __yield_buffer(self, data_path, buffer_size=int(1e5)):
        sample_idx, max_dim, pos_num = 0, 0, 0
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
                ctx_idx, ctx_value = zip(*[[float(j) for j in i.split(':')] for i in context.split(' ')])
                item_array = np.zeros((2, pos_num), dtype=np.float32)
                item_array[0, :] = item_idx
                item_array[1, :] = item_value
                ctx_array = np.zeros((2, self.max_ctx_num), dtype=np.float32)
                ctx_array[0, :len(ctx_idx)] = ctx_idx
                ctx_array[1, :len(ctx_value)] = ctx_value
                tmp_max_dim = max(ctx_idx)
                if tmp_max_dim > max_dim:
                    max_dim = tmp_max_dim
                buf.append((b'citem_%d'%sample_idx, item_array.tobytes(), b'ctx_%d'%sample_idx ,ctx_array.tobytes(), max_dim, pos_num))
                sample_idx += 1
                if sample_idx % buffer_size == 0:
                    yield buf
                    buf.clear()
            yield buf

    def __len__(self):
        return self.length

    #@profile
    def __getitem__(self, idx):  # idx = 10*context_idx + pos
        if not self.test_flag:
            context_idx = idx//self.pos_num
            pos = int(idx%self.pos_num)
            with self.env.begin(write=False) as txn:
                item_array = np.frombuffer(txn.get(b'citem_%d'%context_idx), dtype=np.float32)
                ctx_array = np.frombuffer(txn.get(b'ctx_%d'%context_idx), dtype=np.float32)
                #print(item_array.shape, ctx_array.shape)
                item_idx = item_array[pos]
                flag = item_array[self.pos_num + pos]
                item = np.frombuffer(txn.get(b'item_%d'%item_idx), dtype=np.int32)
                ctx_idx = ctx_array[:self.max_ctx_num].copy()  # context
                ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
            pos += 1
        else:
            context_idx = int(idx)//self.item_num
            item_idx = int(idx)%self.item_num 
            pos = 0
            with self.env.begin(write=False) as txn:
                item_array = np.frombuffer(txn.get(b'citem_%d'%context_idx), dtype=np.float32)
                ctx_array = np.frombuffer(txn.get(b'ctx_%d'%context_idx), dtype=np.float32)
                flag = -1
                item = np.frombuffer(txn.get(b'item_%d'%item_idx), dtype=np.int32)
                ctx_idx = ctx_array[:self.max_ctx_num].copy()  # context
                ctx_value = ctx_array[self.max_ctx_num:].copy()  # context
        if self.tr_max_dim > 0:
            ctx_idx[ctx_idx > self.tr_max_dim] = 0
            ctx_value[ctx_idx > self.tr_max_dim] = 0
        #return {'context':data, 'item':item, 'label':flag, 'pos':pos, 'item_idx':item_idx, 'value':value}  # pos \in {1,2,...9,10}, 0 for no-position
        #print(data.shape, item.shape, flag, pos, item_idx, value.shape)
        return ctx_idx, item, flag, np.array([pos]), item_idx, ctx_value  # pos \in {1,2,...9,10}, 0 for no-position


    def get_max_dim(self):
        return self.max_dim

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

    #@profile
    def main(dataset):
        device='cuda:0'
        #data_loader = DataLoader(dataset, batch_size=4096, num_workers=0, collate_fn=collate_fn_for_dssm, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=4096, num_workers=0,)# shuffle=True)
    
        pbar = tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
        for i, data_pack in enumerate(pbar):
            context, item, target, pos, _, value = data_pack
            context, item, target, pos, value = context.to(device, torch.long), item.to(device, torch.long), target.to(device, torch.float), pos.to(device, torch.long), value.to(device, torch.float)
            #print(context[30:32], item[30:32], target[30:32], pos[30:32], value[30:32])
            print(context.size(), item.size(), target.size(), pos.size(), value.size())
            break
            #print(idx, data, label, pos)

    dataset = PositionDataset(dataset_path=sys.argv[1], data_prefix='va', rebuild_cache=True, tr_max_dim=-1, test_flag=False)
    #dataset = PositionDataset(dataset_path=sys.argv[1], data_prefix='va', rebuild_cache=False, tr_max_dim=-1, test_flag=False)
    main(dataset)
