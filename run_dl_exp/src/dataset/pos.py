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
    def __init__(self, dataset_path=None, data_prefix='tr', rebuild_cache=False, test_flag=False):
        self.min_feat_cnt = 0  # drop feature if it appears less than {min_feat_cnt} times.
        self.test_flag = test_flag

        # build cache
        item_path = os.path.join(dataset_path, 'item.ffm')
        assert Path(item_path).exists(), "%s does not exist!"%item_path
        data_path = os.path.join(dataset_path, data_prefix + '.ffm')
        assert Path(data_path).exists(), "%s does not exist!"%data_path
        cache_path = os.path.join(dataset_path, data_prefix + '.lmdb')
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(item_path, data_path, cache_path)

        # read data
        print('Reading data from %s.'%(cache_path))
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            #self.max_dim = np.frombuffer(txn.get(b'max_dim'), dtype=np.int32)[0] + 1  # idx from 0 to max_dim_in_svmfile, 0 for padding
            self.item = np.frombuffer(txn.get(b'items'), dtype=np.float32)
            self.item_num = self.item.shape[0]
            self.pos_num = int(txn.get(b'pos_num'))
            self.max_cntx_col = int(txn.get(b'max_cntx_col'))
            self.max_item_col = int(txn.get(b'max_item_num'))
            self.length = self.pos_num*(txn.stat()['entries'] - 4)//2 if not self.test_flag else self.item_num*(txn.stat()['entries'] - 4)//2
            print('Totally %d items, %d dims, %d positions, %d samples'%(self.item_num, self.max_dim, self.pos_num, self.length))
    
    def __get_feat_mapper(self, item_path, data_path):
        item_num = 0
        position_num = 0
        max_item_col = 0
        max_cntx_col = 0

        print('Create cache:')
        feat_cnts = defaultdict(lambda: defaultdict(int))  # {field_id:str, feature:str, cnt:int}
        with open(item_path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Counting item features')
            for line in pbar:
                feats = line.strip().split(' ')
                item_num += 1
                max_item_col = max(max_item_col, len(feats))
                for feat in feats:
                    field_idx, feat_idx, feat_value = feat.split(':')
                    #field += '_item'
                    feat_cnts[int(field_idx)][int(feat_idx)] += 1
        item_field_num = len(feat_cnts.items())

        with open(data_path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Counting cntx features')
            for line in pbar:
                labels, values = line.strip().split(' ', 1)
                labels = labels.split(',')
                feats = feats.split(' ')
                position_num = max(position, len(labels))
                max_cntx_col = max(max_cntx_col, len(feats))
                for feat in feats:
                    field_idx, feat_idx, feat_value = feat.split(':')
                    #field += '_cntx'
                    feat_cnts[int(field_idx)+item_field_num][int(feat_idx)] += 1

        feat_mapper = {field: {feat for feat, c in cnt.items() if c >= self.min_threshold} for field, cnt in feat_cnts.items()}  # {field_id, feature_set} 
        #feat_mapper = {field: {feat: idx for idx, feat in enumerate(feats)} for field, feats in feat_mapper.items()}  # {field_id, feature_id, feature}
        #field_dims = np.zeros(len(feat_mapper.items()))
        field_offsets = defaultdict(int)
        tmp = 0
        for field, feat in feat_mapper.items():
            field_offsets[field] = tmp
            tmp += len(feat)

        return feat_mapper, item_num, item_field_num, position_num, max_item_col, max_cntx_col, field_offsets 

    def __build_cache(self, item_path, data_path, cache_path):
        feature_mapper, item_num, item_field_num, pos_num, max_item_col, max_cntx_col, field_offsets = self.__get_feature_mapper(item_path, data_path)
        #field_offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            items = np.zeros((item_num, 3, max_item_col), dtype=np.float32)
            with open(item_path, 'r') as f:
                pbar = tqdm(f, mininterval=1, smoothing=0.1)
                pbar.set_description('Setup lmdb for item')
                for i, line in enumerate(pbar):
                    line = line.strip()
                    feats = sorted(line.split(' '), key=lambda x:x.split(':')[0])
                    for j, feat in enumerate(feats):
                        field_idx, feat_idx, feat_value = feat.split(':')
                        items[i][0][j] = int(feat_idx) + 1 + field_offsets[int(field_idx)] # feat_idx
                        items[i][1][j] = float(feat_value)  # feat_value
                        items[i][2][j] = int(field_idx)  # feat_value
                
            for buf in self.__yield_buffer(data_path, items, feature_mapper, item_field_num, pos_num, max_item_col, max_cntx_col):
                with env.begin(write=True) as txn:
                    for item_key, item_array, ctx_key, ctx_array, max_dim_buf, pos_num[0] in buf:
                        txn.put(item_key, item_array)
                        txn.put(ctx_key, ctx_array)
        
            with env.begin(write=True) as txn:
                txn.put(b'items', items.tobytes())
                #txn.put(b'item_num', b'%d'%item_num)
                txn.put(b'pos_num', b'%d'%pos_num)
                txn.put(b'max_item_col', b'%d'%max_item_col)
                txn.put(b'max_cntx_col', b'%d'%max_cntx_col)
                txn.put(b'field_dims', field_dims.tobytes())

    def __yield_buffer(self, data_path, items, feature_mapper, item_field_num, pos_num, max_item_col, max_cntx_col, buffer_size=int(5e5)):
        sample_idx = 0
        buf = list()
        with open(data_path, 'r') as fd:
            pbar = tqdm(fd, mininterval=1, smoothing=0.1)
            pbar.set_description('Setup lmdb for context')
            for line in pbar:
                line = line.strip()
                labels, feats = line.split(' ', 1)
                feat_array = np.zeros((3, max_item_col+max_cntx_col))
                feats = sorted(feats.split(' '), key=lambda x: x.split(':')[0])
                feats_num = len(feats)
                for i, feat in enumerate(feats):
                    field_idx, feat_idx, feat_value = feat.split(':')
                    feat_array[0, i] = int(feat_idx) + 1 + field_offsets[int(field_idx)+item_field_num]
                    feat_array[1, i] = float(feat_value)
                    feat_array[2, i] = int(field_idx) + item_field_num
                
                #item, click = zip(*[[int(j) for j in i.split(':')[:2]] for i in labels.strip().split(',')])
                #field, feat, feat_value = zip(*[[j for j in i.split(':')] for i in context.split(' ')])
                for i, label in enumerate(labels.strip().split(',')):
                    item, click, _ = label.split(':')
                    for j in items[int(item), :]:
                        feat_array[0, i] =  
                        feat_array[1, i] = float(feat_value)
                        feat_array[2, i] = int(field_idx) + item_field_num

                item_array = np.zeros((2, pos_num), dtype=np.float32)
                item_array[0, :] = item
                item_array[1, :] = click

                ctx_array = np.zeros((2, max_cntx_col), dtype=np.float32)
                ctx_array[0, :len(feat)] = [feature_mapper[i+'_cntx'] for i in feat]
                ctx_array[1, :len(cntx_value)] = [float(i) for i in feat_value]

                buf.append((b'citem_%d'%sample_idx, item_array.tobytes(), b'cntx_%d'%sample_idx ,ctx_array.tobytes()))
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
                ctx_array = np.frombuffer(txn.get(b'cntx_%d'%context_idx), dtype=np.float32)
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
