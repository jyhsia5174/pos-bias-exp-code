import torch
import os
import time
import tqdm
import numpy as np
import sys
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils

from src.dataset.position import PositionDataset
from src.dataset.a9a import A9ADataset
from src.model.lr import LogisticRegression
from src.model.bilr import BiLogisticRegression
from src.model.extlr import ExtLogisticRegression
from src.model.dssm import DSSM
from src.model.bidssm import BiDSSM
from utility import recommend

np.random.seed(0)

def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def hook(self, input, output):
    tmp = torch.sigmoid(output.data).flatten().tolist()
    ratio = [tmp[0]]
    for i in range(1, 10):
        ratio.append(tmp[i+1]/tmp[i])
    print(tmp)
    print(ratio, np.mean(ratio[1:]))

def collate_fn_for_lr(batch):
    data = [torch.LongTensor(np.hstack((i['item'], i['context']))) for i in batch]
    label = [i['label'] for i in batch]
    pos = [i['pos'] for i in batch]
    #if 0 in pos:
    #    print("The position padding_idx occurs!")
    #data.sort(key=lambda x: len(x), reverse=True)
    #data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, torch.FloatTensor(label), torch.FloatTensor(pos).unsqueeze(-1)

def collate_fn_for_dssm(batch):
    context = [torch.LongTensor(i['context']) for i in batch]
    item = [torch.LongTensor(i['item']) for i in batch]
    label = [i['label'] for i in batch]
    pos = [i['pos'] for i in batch]
    #if 0 in pos:
    #    print("The position padding_idx occurs!")
    #data.sort(key=lambda x: len(x), reverse=True)
    #data_length = [len(sq) for sq in data]
    item = rnn_utils.pad_sequence(item, batch_first=True, padding_value=0)
    context = rnn_utils.pad_sequence(context, batch_first=True, padding_value=0)
    return context, item, torch.FloatTensor(label), torch.FloatTensor(pos).unsqueeze(-1)

def get_dataset(name, path, data_prefix, rebuild_cache, max_dim=-1, test_flag=False):
    if name == 'pos':
        return PositionDataset(path, data_prefix, rebuild_cache, max_dim, test_flag)
    if name == 'a9a':
        return A9ADataset(path, training)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    input_dims = dataset.max_dim
    if name == 'lr':
        return LogisticRegression(input_dims)
    elif name == 'bilr':
        return BiLogisticRegression(input_dims, 10)
    elif name == 'extlr':
        return ExtLogisticRegression(input_dims, 10)
    elif name == 'dssm':
        return DSSM(input_dims)
    elif name == 'bidssm':
        return BiDSSM(input_dims, 10)
    else:
        raise ValueError('unknown model name: ' + name)


def pred(model, data_loader, device, model_name):
    num_of_user = data_loader.batch_size//1055 
    k = 10
    bids = np.random.gamma(20, 0.4, 1055)
    gbids = torch.tensor(bids).expand(num_of_user, -1).flatten().to(device)
    res = np.empty(num_of_user*k, dtype=np.int32)

    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad(), open('tmp.pred', 'w') as fp:
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            if 'bilr' == model_name or 'extlr' == model_name:
                data, target, pos = tmp
                data, target, pos = data.to(device, torch.long), target.to(device, torch.float), pos.to(device, torch.long)
                y = model(data, pos)
            elif model_name == 'dssm':
                context, item, target, pos = tmp
                context, item, target = context.to(device, torch.long), item.to(device, torch.long), target.to(device, torch.float)
                y = model(context, item)
            elif model_name == 'bidssm':
                context, item, target, pos = tmp
                context, item, target, pos = context.to(device, torch.long), item.to(device, torch.long), target.to(device, torch.float), pos.to(device, torch.long)
                y = model(context, item, pos)
            else:
                data, target, pos = tmp
                data, target = data.to(device, torch.long), target.to(device, torch.float)
                y = model(data)
            out = y*gbids
            #print(out.shape)
            recommend.get_top_k_by_greedy(out.cpu().numpy(), num_of_user, 1055, k, res)
            _res = res.reshape(num_of_user, k)
            for r in range(num_of_user):
                tmp = ['%d:%.4f'%(ad, bids[ad]) for ad in _res[r, :]]
                fp.write('%s\n'%(' '.join(tmp)))
            #if i>=2:
            #    break
            #targets.extend(torch.flatten(target.to(torch.int)).tolist())
            #predicts.extend([j*2. for j in torch.flatten(y).tolist()])
    #return predicts


def main(dataset_name,
         dataset_path,
         model_name,
         model_path,
         #epoch,
         #learning_rate,
         batch_size,
         #weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    if model_name == 'dssm' or model_name == 'bidssm':
        collate_fn = collate_fn_for_dssm 
    else:
        collate_fn = collate_fn_for_lr 
    train_dataset = get_dataset(dataset_name, dataset_path, 'trva', False)
    valid_dataset = get_dataset(dataset_name, dataset_path, 'gt', False, train_dataset.get_max_dim() - 1, True)
    #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)
    #model = get_model(model_name, train_dataset).to(device)
    #criterion = torch.nn.BCELoss()
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #for epoch_i in range(epoch):
    #    train(model, optimizer, train_data_loader, criterion, device, model_name)
    #    auc, logloss = test(model, valid_data_loader, device, model_name)
    #    print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)
    model = torch.load(model_path)
    pred(model, valid_data_loader, device, model_name)
    #print('test auc:', auc)
    #model_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), 'bs-'+str(batch_size)])
    #torch.save(model, f'{save_dir}/{model_name}.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='pos')
    parser.add_argument('--dataset_path', help='the path that contains item.svm, va.svm, tr.svm')
    parser.add_argument('--model_name', default='dssm')
    parser.add_argument('--model_path', help='the model path')
    #parser.add_argument('--epoch', type=int, default=10)
    #parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1055*20)
    #parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0')
    #parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='tmp')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.model_path,
         #args.epoch,
         #args.learning_rate,
         args.batch_size,
         #args.weight_decay,
         args.device,
         args.save_dir)
