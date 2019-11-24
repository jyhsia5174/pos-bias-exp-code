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

def collate_fn(batch):
    data = [torch.LongTensor(i['data']) for i in batch]
    label = [i['label'] for i in batch]
    pos = [i['pos'] for i in batch]
    #data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data, data_length, torch.FloatTensor(label), torch.FloatTensor(pos).unsqueeze(-1)

def get_dataset(name, path, n_feature, training):
    if name == 'pos':
        return PositionDataset(path, n_feature, training)
    if name == 'a9a':
        return A9ADataset(path, training)
    else:
        raise ValueError('unknown dataset name: ' + name)


def get_model(name, dataset):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    input_dims = dataset.n_feature
    if name == 'lr':
        return LogisticRegression(input_dims)
    #elif name == 'bilr':
    #    return BiLogisticRegressionModel(input_dims-10, 10)
    else:
        raise ValueError('unknown model name: ' + name)


def train(model, optimizer, data_loader, criterion, device, log_interval=1000):
    model.train()
    total_loss = 0
    for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
        data, data_len, target, pos = tmp
        del tmp
        data, target = data.to(device, torch.long), target.to(device, torch.float)
        y = model(data)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0


def test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
            data, data_len, target, pos = tmp
            #print(data[-10:], target[-10:], pos[-10:])
            #sys.exit(0)
            data, target = data.to(device, torch.long), target.to(device, torch.float)
            y = model(data)
            targets.extend(torch.flatten(target.to(torch.int)).tolist())
            predicts.extend(torch.flatten(y).tolist())
    #print(targets[:10], predicts[:10])
    #print(predicts[np.argmax(targets)])
    #print(min(targets), min(predicts))
    #print(set(targets))
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)


def main(dataset_name,
         dataset_path,
         model_name,
         epoch,
         learning_rate,
         batch_size,
         weight_decay,
         device,
         save_dir):
    device = torch.device(device)
    train_dataset = get_dataset(dataset_name, dataset_path, 0, True)
    valid_dataset = get_dataset(dataset_name, dataset_path, train_dataset.get_n_feature(), False)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, collate_fn=collate_fn)
    model = get_model(model_name, train_dataset).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, device)
        auc, logloss = test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc, 'logloss:', logloss)
    #auc = test(model, valid_data_loader, device)
    #print('test auc:', auc)
    torch.save(model, f'{save_dir}/{model_name}.pt')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='pos')
    parser.add_argument('--dataset_path', help='the path that contains item.svm, va.svm, tr.svm')
    parser.add_argument('--model_name', default='lr')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--device', default='cuda:0')
    #parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='tmp')
    args = parser.parse_args()
    main(args.dataset_name,
         args.dataset_path,
         args.model_name,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir)
