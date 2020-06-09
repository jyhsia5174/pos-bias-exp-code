import torch
import os
import time
import tqdm
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils

#from src.dataset.ffmdl import FFMDataset
from src.dataset.ffmdl_batch_in_mem import FFMDataset
from src.model.ffm import FieldAwareFactorizationMachineModel as FFM
from src.model.dfm import DeepFactorizationMachineModel as DFM
#from utility import recommend


def collate_fn(batch):
    if len(batch[0]) == 5:
        res = []
        for i in range(len(batch[0])):
            res.append(torch.cat([torch.as_tensor(b[i]) for b in batch], dim=0))
        return tuple(res)
    elif len(batch[0]) == 2:
        res1 = []
        for i in range(len(batch[0][0])):
            res1.append(torch.cat([torch.as_tensor(b[0][i]) for b in batch], dim=0))
        res2 = []
        for i in range(len(batch[0][1])):
            res2.append(torch.cat([torch.as_tensor(b[1][i]) for b in batch], dim=0))
        return tuple(res1), tuple(res2)
    else:
        raise
    return

def precision_at_k(y_true, y_score, ks):
    p = np.zeros(len(ks))
    for i in range(y_true.shape[0]):
        sorted_id = np.argsort(y_score[i, :])[::-1]
        for j, k in enumerate(ks): 
            p[j] += (y_true[i, sorted_id[:k]].sum()/k)
    return p

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_dims(t):
    return t.view(tuple(-1 if i==0 else _s for i, _s in enumerate(t.size()[1:])))

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

def get_dataset(name, path, data_prefix, rebuild_cache, tr_field_dims=None, read_flag=0):
    if name == 'ffmdl':
        return FFMDataset(path, data_prefix, rebuild_cache, tr_field_dims, read_flag)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, field_dims, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    if name == 'ffm':
        return FFM(field_dims, embed_dim)
    elif name == 'dfm':
        return DFM(field_dims, embed_dim=embed_dim, mlp_dims=(embed_dim, embed_dim), dropout=0.2)
    else:
        raise ValueError('unknown model name: ' + name)


def model_helper(data_pack, model, model_name, device):
    context, item, target, pos, _ = data_pack
    data = torch.cat((context, item), dim=-1)
    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
    y = model(data[:, 0, :].to(torch.long), data[:, 1, :].to(torch.long), data[:, 2, :])

    return y, target

def train(model, optimizer, data_loader, criterion, device, model_name, log_interval=1000):
    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
    for i, data_pack in enumerate(pbar):
        y, target = model_helper(data_pack, model, model_name, device)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            closs = total_loss/log_interval
            pbar.set_postfix(loss=closs)
            total_loss = 0
    return loss.item()

def bpr_train(model, optimizer, data_loader, device, model_name):
    model.train()
    total_loss = 0
    pbar = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
    for i, (pos_data_pack, neg_data_pack) in enumerate(pbar):
        pos_y, target = model_helper(pos_data_pack, model, model_name, device)
        neg_y, _ = model_helper(neg_data_pack, model, model_name, device)

        loss = - (pos_y - neg_y).sigmoid().log().mean()
        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        closs = total_loss/(i+1)
        pbar.set_postfix(bpr_loss=closs)

    return closs

def test(model, data_loader, device, model_name):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)):
            y, target = model_helper(tmp, model, model_name, device)
            y = torch.sigmoid(y)  # 1,2,3//4,5
            #num_of_user = y.size()[0]//10
            targets.extend(torch.flatten(target.to(torch.int)).tolist())
            predicts.extend(torch.flatten(y).tolist())
    #return roc_auc_score(targets, predicts), log_loss(targets, predicts)
    return 0, log_loss(targets, predicts)

def test_ranking(model, data_loader, device, model_name, item_num, eva_k):
    model.eval()
    #targets, predicts = list(), list()
    ks = [5, 10, 20, 40]
    p = np.zeros(len(ks))
    count = 0
    #ndcg = np.zeros(len(ks))
    with torch.no_grad():
        #for i, tmp in enumerate(tqdm.tqdm(va_data_loader, smoothing=0, mininterval=1.0, ncols=100)):
        pbar = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
        #for i, (pos_data_pack, neg_data_pack) in enumerate(pbar):
        for i, data_pack in enumerate(pbar):
            y, target = model_helper(data_pack, model, model_name, device)
            targets = np.array(torch.flatten(target.to(torch.int)).tolist()).reshape(-1, item_num)
            predicts = np.array(torch.flatten(y).tolist()).reshape(-1, item_num)
            count+=targets.shape[0]
            p += precision_at_k(targets, predicts, ks)
    return p/count
            #ndcg += ndcg_score(targets, predicts, eva_k)

def pred(model, data_loader, device, model_name):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)):
            y, _ = model_helper(tmp, model, model_name, device)
            y = torch.sigmoid(y)
            predicts.extend(torch.flatten(y).tolist())
    return predicts

#def pred(model, data_loader, device, model_name, item_num):
#    num_of_pos = 10
#    res = np.empty(data_loader.batch_size//item_num*num_of_pos, dtype=np.int32)
#    rngs = [np.random.RandomState(seed) for seed in [0,3,4,5,6]]
#    bids = np.empty((len(rngs), item_num)) 
#    for i, rng in enumerate(rngs):
#        bids[i, :] = rng.gamma(10, 0.4, item_num)
#    bids = torch.tensor(bids).to(device)
#
#    model.eval()
#    targets, predicts = list(), list()
#    with torch.no_grad():
#        fs = list()
#        for j in range(len(rngs)):
#            fs.append(open(os.path.join('tmp.pred.%d'%j), 'w'))
#        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)):
#            y, target = model_helper(tmp, model, model_name, device, mode='wops')
#            num_of_user = y.size()[0]//item_num
#            for j in range(len(rngs)):
#                fp = fs[j]
#                out = y*(bids[j, :].repeat(num_of_user))
#                #recommend.get_top_k_by_greedy(out.cpu().numpy(), num_of_user, item_num, num_of_pos, res[:num_of_user*num_of_pos])
#                _res = res[:num_of_user*num_of_pos].reshape(num_of_user, num_of_pos)
#                for r in range(num_of_user):
#                    tmp = ['%d:%.4f:%0.4f'%(ad, y[r*item_num+ad], bids[j, ad]) for ad in _res[r, :]]
#                    #tmp = ['%d:%.4f'%(ad, bids[j, ad]) for ad in _res[r, :]]
#                    fp.write('%s\n'%(' '.join(tmp)))


def main(dataset_name,
         train_part,
         valid_part,
         dataset_path,
         flag,
         model_name,
         model_path,
         epoch,
         learning_rate,
         batch_size,
         embed_dim,
         weight_decay,
         eva_k,
         device,
         save_dir,
         ps):
    mkdir_if_not_exist(save_dir)
    device = torch.device(device)
    if flag == 'train':
        train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, train_dataset.field_dims, 0)
        #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
        #valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True, collate_fn=collate_fn)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, collate_fn=collate_fn)
        model = get_model(model_name, train_dataset.field_dims, embed_dim).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_file_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), 'bs-'+str(batch_size), 'k-'+str(embed_dim), train_part])
        with open(os.path.join(save_dir, model_file_name+'.log'), 'w') as log:
            for epoch_i in range(epoch):
                tr_logloss = train(model, optimizer, train_data_loader, criterion, device, model_name)
                va_auc, va_logloss = test(model, valid_data_loader, device, model_name)
                print('epoch:%d\ttr_logloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f'%(epoch_i, tr_logloss, va_auc, va_logloss))
                log.write('epoch:%d\ttr_logloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f\n'%(epoch_i, tr_logloss, va_auc, va_logloss))
        torch.save(model, f'{save_dir}/{model_file_name}.pt')
    if flag == 'bpr_train':
        pos_train_dataset = get_dataset(dataset_name, dataset_path, train_part, False, None, 0)
        neg_train_dataset = get_dataset(dataset_name, dataset_path, train_part, False, None, 2)
        train_dataset = SimDataset(pos_train_dataset, neg_train_dataset)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, pos_train_dataset.field_dims, 1)
        #valid_dataset = torch.utils.data.RandomSampler(valid_dataset, num_samples=len(valid_dataset)//1000)
        #neg_valid_dataset = get_dataset(dataset_name, dataset_path, train_part, False, pos_train_dataset.field_dims, 1)
        #valid_dataset = SimDataset(pos_valid_dataset, neg_valid_dataset)
        #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
        #valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True, collate_fn=collate_fn)
        valid_data_loader = DataLoader(valid_dataset, batch_size=50, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        model = get_model(model_name, pos_train_dataset.field_dims, embed_dim).to(device)
        #criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_file_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), 'bs-'+str(batch_size), 'k-'+str(embed_dim), train_part])
        with open(os.path.join(save_dir, model_file_name+'.log'), 'w') as log:
            for epoch_i in range(epoch):
                #sample_idxes = np.random.randint(len(valid_dataset), size=1000*valid_dataset.item_num)
                #valid_subset = torch.utils.data.Subset(valid_dataset, sample_idxes)
                #valid_data_loader = DataLoader(valid_subset, batch_size=batch_size, num_workers=10, pin_memory=True)
                tr_logloss = bpr_train(model, optimizer, train_data_loader, device, model_name)
                if (epoch_i+1)%3 == 0:
                    va_patk = test_ranking(model, valid_data_loader, device, model_name, valid_dataset.item_num, eva_k)
                    #print('epoch:%d\ttr_bprloss:%.6f\tva_p@%d:%.6f\tva_ndcg@%d:%.6f'%(epoch_i, tr_logloss, eva_k, va_patk, eva_k, va_ndcg))
                    print('epoch:%d\ttr_bprloss:%.6f\tva_p@[5,10,20,40]:%s'%(epoch_i, tr_logloss,','.join(['%.6f'%p for p in va_patk])))
                    log.write('epoch:%d\ttr_bprloss:%.6f\tva_p@[5,10,20,40]:%s\n'%(epoch_i, tr_logloss,','.join(['%.6f'%p for p in va_patk])))
        torch.save(model, f'{save_dir}/{model_file_name}.pt')
    #elif flag == 'pred':
    #    #train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
    #    valid_dataset = get_dataset(dataset_name, dataset_path, train_part, False)#, train_dataset.field_dims, 0)
    #    #item_num = valid_dataset.get_item_num()
    #    #refine_batch_size = int(batch_size//item_num*item_num)  # batch_size should be a multiple of item_num 
    #    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
    #    model = torch.load(model_path).to(device)
    #    pred(model, valid_data_loader, device, model_name, item_num)
    elif flag == 'test':
        train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, train_dataset.field_dims, 0)
        #valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, collate_fn=collate_fn)
        #print(device)
        model = torch.load(model_path, map_location=device)
        va_auc, va_logloss = test(model, valid_data_loader, device, model_name)
        print("model logloss auc")
        print("%s %.6f %.6f"%(model_name, va_logloss, va_auc))
        #pred(model, valid_data_loader, device, model_name, item_num)
    else:
        raise ValueError('Flag should be "train"/"pred"/"test_auc"!')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='ffmdl')
    parser.add_argument('--train_part', default='tr')
    parser.add_argument('--valid_part', default='va')
    parser.add_argument('--dataset_path', help='the path that contains item.svm, va.svm, tr.svm trva.svm')
    parser.add_argument('--flag', default='train')
    parser.add_argument('--model_name', default='ffm')
    parser.add_argument('--model_path', default='', help='the path of model file')
    parser.add_argument('--epoch', type=float, default=30.)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=float, default=500.)
    parser.add_argument('--embed_dim', type=float, default=32.)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--eva_k', type=int, default=5)
    parser.add_argument('--device', default='cuda:0', help='format like "cuda:0" or "cpu"')
    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--ps', default='wps')
    args = parser.parse_args()
    main(args.dataset_name,
         args.train_part,
         args.valid_part,
         args.dataset_path,
         args.flag,
         args.model_name,
         args.model_path,
         int(args.epoch),
         args.learning_rate,
         int(args.batch_size),
         int(args.embed_dim),
         args.weight_decay,
         args.eva_k,
         args.device,
         args.save_dir,
         args.ps)

