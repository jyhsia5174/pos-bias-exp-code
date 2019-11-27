import torch
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

def hook(self, input, output):
    tmp = torch.sigmoid(output.data).flatten().tolist()
    ratio = [tmp[0]]
    for i in range(1, 10):
        ratio.append(tmp[i+1]/tmp[i])
    print(tmp)
    print(ratio, np.mean(ratio))

def collate_fn(batch):
    data = [torch.LongTensor(i['data']) for i in batch]
    label = [i['label'] for i in batch]
    pos = [i['pos'] for i in batch]
    if 0 in pos:
        print("The position padding_idx occurs!")
    #data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_length, torch.FloatTensor(label), torch.FloatTensor(pos).unsqueeze(-1)

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
    else:
        raise ValueError('unknown model name: ' + name)


def pred(model, data_loader, device, model_name):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad(), open('tmp.pred', 'w') as fp:
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            data, data_len, target, pos = tmp
            if 'bi' in model_name:
                data, target, pos= data.to(device, torch.long), target.to(device, torch.float), pos.to(device, torch.long)
                y = model(data, pos)
            else:
                data, target = data.to(device, torch.long), target.to(device, torch.float)
                y = model(data)
            for p in torch.flatten(y).tolist():
                fp.write('%.4f\n'%(p*2.))
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
    train_dataset = get_dataset(dataset_name, dataset_path, 'trva', False)
    valid_dataset = get_dataset(dataset_name, dataset_path, 'va', False, train_dataset.get_max_dim() - 1, True)
    #train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, collate_fn=collate_fn)
    model = get_model(model_name, train_dataset).to(device)
    #criterion = torch.nn.BCELoss()
    #optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #for epoch_i in range(epoch):
    #    train(model, optimizer, train_data_loader, criterion, device, model_name)
    #    auc, logloss = test(model, valid_data_loader, device, model_name)
    #    print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)
    model.load_state_dict(torch.load(model_path))
    targets = pred(model, valid_data_loader, device)
    print(targets)
    #print('test auc:', auc)
    #model_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), 'bs-'+str(batch_size)])
    #torch.save(model, f'{save_dir}/{model_name}.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='pos')
    parser.add_argument('--dataset_path', help='the path that contains item.svm, va.svm, tr.svm')
    parser.add_argument('--model_name', default='bilr')
    parser.add_argument('--model_path', help='the model path')
    #parser.add_argument('--epoch', type=int, default=10)
    #parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=8192)
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
