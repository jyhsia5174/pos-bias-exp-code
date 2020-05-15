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
from src.dataset.yh import YHDataset
from src.dataset.a9a import A9ADataset
from src.model.lr import LogisticRegression
from src.model.bilr import BiLogisticRegression
from src.model.extlr import ExtLogisticRegression
from src.model.dssm import DSSM
from src.model.bidssm import BiDSSM
from src.model.extdssm import ExtDSSM
from src.model.ffm import FFM
from src.model.biffm import BiFFM
from src.model.extffm import ExtFFM
from src.model.xdfm import ExtremeDeepFactorizationMachineModel
from src.model.bixdfm import BiExtremeDeepFactorizationMachineModel
from src.model.extxdfm import ExtExtremeDeepFactorizationMachineModel
from src.model.dfm import DeepFactorizationMachineModel
from src.model.dcn import DeepCrossNetworkModel
from src.model.naive_gan import Generator as G, Discriminator as D
from utility import recommend


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def merge_dims(t):
    return t.view(tuple(-1 if i==0 else _s for i, _s in enumerate(t.size()[1:])))

def hook(self, input, output):
    tmp = torch.sigmoid(output.data).flatten().tolist()
    ratio = [tmp[0]]
    for i in range(1, 10):
        ratio.append(tmp[i+1]/tmp[i])
    print(tmp)
    print(ratio, np.mean(ratio[1:]))

def get_dataset(name, path, data_prefix, rebuild_cache, max_dim=-1, test_flag='0'):
    if name == 'pos':
        #return PositionDataset(path, data_prefix, True, max_dim, test_flag)
        return PositionDataset(path, data_prefix, rebuild_cache, max_dim, test_flag)
    if name == 'a9a':
        return A9ADataset(path, training)
    if name == 'yh':
        return YHDataset(path, data_prefix, rebuild_cache, max_dim, test_flag)
    else:
        raise ValueError('unknown dataset name: ' + name)

def get_model(name, dataset, embed_dim):
    """
    Hyperparameters are empirically determined, not opitmized.
    """
    input_dims = dataset.max_dim
    if name == 'ffm':
        return FFM(input_dims, embed_dim)
    elif name == 'biffm':
        return BiFFM(input_dims, dataset.pos_num, embed_dim)
    elif name == 'extffm':
        return ExtFFM(input_dims, dataset.pos_num, embed_dim)
    else:
        raise ValueError('unknown model name: ' + name)


def model_helper(data_pack, model, device, label=None):
    context, item, target, _, _, value, _ = data_pack
    context, item, target, value = context.to(device, torch.long), item.to(device, torch.long), target.to(device, torch.float), value.to(device, torch.float)
    y = model(context, item, None, value)
    if label is not None:
        if type(label) in (int, float):
            target = torch.full(target.szie(), label)
        else:
            raise TypeError()
    return y, target

def train(model, optimizer, data_loader, criterion, device, model_name):
    model.train()
    c = 0
    total_loss = 0
    pbar = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
    for i, tmp in enumerate(pbar):
        c+=1
        y, target = model_helper(tmp, model, device)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/c)
    return total_loss/c

def data_helper(data_pack, device):
    context, item, target, _, _, value, obs_label = data_pack
    context, item, target, value, obs_label = context.to(device, torch.long), item.to(device, torch.long), target.to(device, torch.float), value.to(device, torch.float), obs_label.to(device, torch.float)
    return context, item, target, value, obs_label

def obs_train(model, optimizer, data_loader, criterion, device):
    model.train()
    c = 0
    total_loss = 0
    pbar = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)
    for i, tmp in enumerate(pbar):
        c+=1
        cntx, item, y, val, v = data_helper(tmp, device)
        loss = criterion(model(cntx, item, y, None, val), v)
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss/c)
    return total_loss/c

def new_train(model, imp_model, optimizer, data_loader, imp_data_loader, imp_type, device, omega):
    model.train()
    c = 0
    total_loss1 = 0
    total_loss2 = 0
    imp_data_loader = iter(imp_data_loader)
    criterion = torch.nn.BCEWithLogitsLoss()
    imp_criterion = torch.nn.MSELoss(reduction='none')
    pbar = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)

    for i, batch in enumerate(pbar):
        c+=1
        imp_batch = next(imp_data_loader)
        cntx, item, y, val, _ = data_helper(batch, device)
        imp_cntx, imp_item, imp_y, imp_val, _ = data_helper(imp_batch, device)
        weight = torch.zeros_like(imp_y)
        weight[imp_y<0] = 1.

        y_hat = model(cntx, item, None, val)
        loss1 = criterion(y_hat, y)   # bcewithlogit
        
        if imp_type == 'r':
            imp_y = imp_y.fill_(imp_model)
        elif imp_type == 'item-r':
            imp_y = torch.ones_like(imp_y)*imp_model[imp_item.flatten()-1]
        elif imp_type == 'complex':
            imp_y = imp_model(imp_cntx, imp_item, None, imp_val)
        else:
            raise
        imp_y_hat = model(imp_cntx, imp_item, None, imp_val)
        loss2 = (imp_criterion(imp_y_hat, imp_y)*weight).sum() / weight.sum()
        
        loss = loss1 + omega*loss2
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss1 += loss1.item()
        total_loss2 += loss2.item()
        pbar.set_postfix(loss1=total_loss1/c, loss2=total_loss2/c)

    return total_loss1/c, total_loss2/c

def gan_train(generator, discriminator, opt_G, opt_D, rnd_data_loader, det_data_loader, full_data_loader, device, omega, fix_D=True):
    generator.train()
    discriminator.train()
    c = 0
    total_g_loss1 = 0
    total_g_loss2 = 0
    total_d_loss = 0
    
    pbar = tqdm.tqdm(full_data_loader, smoothing=0, mininterval=1.0, ncols=100)
    #pbar = tqdm.tqdm(det_data_loader, smoothing=0, mininterval=1.0, ncols=100)
    rnd_dl = iter(rnd_data_loader)
    #det_dl = iter(det_data_loader)
    #mix_dl = iter(mix_data_loader)

    for i, full_batch in enumerate(pbar):
        c+=1
        try:
            rnd_batch = next(rnd_dl)
        except StopIteration:
            rnd_dl = iter(rnd_data_loader)
            rnd_batch = next(rnd_dl)

        #det_batch = next(det_dl)
        #mix_batch = next(mix_dl)


        # -----------------
        #  Train Generator
        # -----------------
        opt_G.zero_grad()
        
        cntx, item, y, val, _ = data_helper(full_batch, device)
        v = torch.ones_like(y)
        weight = torch.zeros_like(y)  # get the mask for supvised loss
        weight[y>=0] = 1.

        y_hat = generator(cntx, item, None, val)
        criterion_sup = torch.nn.BCEWithLogitsLoss(weight=weight, reduction='sum')
        loss1 = criterion_sup(y_hat, y) / weight.sum()
        ##loss1.backward()

        v_hat = discriminator(cntx, item, torch.sigmoid(y_hat), None, val)
        criterion_gan = torch.nn.BCEWithLogitsLoss(weight=1.-weight, reduction='sum')
        loss2 = criterion_gan(v_hat, v) / (1.-weight).sum()
        #loss2.backward()

        total_g_loss1 += loss1.item()
        total_g_loss2 += loss2.item()
        g_loss = loss1 + omega*loss2
        g_loss.backward()

        opt_G.step()
        
        # ---------------------
        #  Train Discriminator
        # ---------------------

        if not fix_D:
            opt_D.zero_grad()

            fake_loss = criterion_gan(discriminator(cntx, item, torch.sigmoid(y_hat).detach(), None, val), torch.zeros_like(y)) / (1.-weight).sum()

            rnd_cntx, rnd_item, rnd_y, rnd_val, _ = data_helper(rnd_batch, device)
            criterion_gan = torch.nn.BCEWithLogitsLoss()
            rnd_loss = criterion_gan(discriminator(rnd_cntx, rnd_item, rnd_y, None, rnd_val), torch.ones_like(rnd_y))

            d_loss = (rnd_loss + fake_loss) / 2. #+ det_loss)
            total_d_loss += d_loss.item()
            d_loss.backward()
            opt_D.step()

        pbar.set_postfix(g_loss1=total_g_loss1/c, g_loss2=total_g_loss2/c, d_loss=total_d_loss/c)
    return total_g_loss1/c, total_g_loss2/c, total_d_loss/c

def test(model, data_loader, device, model_name, mode='wps'):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)):
            y, target = model_helper(tmp, model, device)
            y = torch.sigmoid(y)
            targets.extend(torch.flatten(target.to(torch.int)).tolist())
            predicts.extend(torch.flatten(y).tolist())
    print("avg pred: %f, avg target: %f"%(np.mean(predicts), np.mean(targets)))
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)

def obs_test(model, data_loader, device, mode='wps'):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)):
            cntx, item, y, val, v = data_helper(tmp, device)
            v_hat = model(cntx, item, y, None, val)
            v_hat = torch.sigmoid(v_hat)
            targets.extend(torch.flatten(v.to(torch.int)).tolist())
            predicts.extend(torch.flatten(v_hat).tolist())
    print("avg pred: %f, avg target: %f"%(np.mean(predicts), np.mean(targets)))
    return roc_auc_score(targets, predicts), log_loss(targets, predicts)

def pred(model, data_loader, device, model_name, item_num):
    num_of_pos = 10
    res = np.empty(data_loader.batch_size//item_num*num_of_pos, dtype=np.int32)
    rngs = [np.random.RandomState(seed) for seed in [0,3,4,5,6]]
    bids = np.empty((len(rngs), item_num)) 
    for i, rng in enumerate(rngs):
        bids[i, :] = rng.gamma(10, 0.4, item_num)
    bids = torch.tensor(bids).to(device)

    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        fs = list()
        for j in range(len(rngs)):
            fs.append(open(os.path.join('tmp.pred.%d'%j), 'w'))
        for i, tmp in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, ncols=100)):
            y, target = model_helper(tmp, model, model_name, device, mode='wops')
            num_of_user = y.size()[0]//item_num
            for j in range(len(rngs)):
                fp = fs[j]
                out = y*(bids[j, :].repeat(num_of_user))
                recommend.get_top_k_by_greedy(out.cpu().numpy(), num_of_user, item_num, num_of_pos, res[:num_of_user*num_of_pos])
                _res = res[:num_of_user*num_of_pos].reshape(num_of_user, num_of_pos)
                for r in range(num_of_user):
                    tmp = ['%d:%.4f:%0.4f'%(ad, y[r*item_num+ad], bids[j, ad]) for ad in _res[r, :]]
                    #tmp = ['%d:%.4f'%(ad, bids[j, ad]) for ad in _res[r, :]]
                    fp.write('%s\n'%(' '.join(tmp)))


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
         device,
         omega,
         save_dir,
         imp_type,
         ps):
    mkdir_if_not_exist(save_dir)
    device = torch.device(device)
    if flag == 'train':
        train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, train_dataset.get_max_dim() - 1)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)
        #model = get_model(model_name, train_dataset, embed_dim).to(device)
        model = G(train_dataset.max_dim, embed_dim).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_file_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), 'bs-'+str(batch_size), 'k-'+str(embed_dim), train_part])
        with open(os.path.join(save_dir, model_file_name+'.log'), 'w') as log:
            for epoch_i in range(epoch):
                tr_logloss = train(model, optimizer, train_data_loader, criterion, device, model_name)
                va_auc, va_logloss = test(model, valid_data_loader, device, model_name, 'wps')
                print('epoch:%d\ttr_logloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f'%(epoch_i, tr_logloss, va_auc, va_logloss))
                log.write('epoch:%d\ttr_logloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f\n'%(epoch_i, tr_logloss, va_auc, va_logloss))
        torch.save(model, f'{save_dir}/{model_file_name}.pt')
    elif flag == 'obs_train':
        train_dataset = get_dataset(dataset_name, dataset_path, train_part, False, -1, '2')
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, -1, '2')
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)
        model = D(train_dataset.max_dim, embed_dim).to(device)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_file_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), 'bs-'+str(batch_size), 'k-'+str(embed_dim), train_part])
        with open(os.path.join(save_dir, model_file_name+'.log'), 'w') as log:
            for epoch_i in range(epoch):
                tr_logloss = obs_train(model, optimizer, train_data_loader, criterion, device)
                va_auc, va_logloss = obs_test(model, valid_data_loader, device, 'wps')
                print('epoch:%d\ttr_logloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f'%(epoch_i, tr_logloss, va_auc, va_logloss))
                log.write('epoch:%d\ttr_logloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f\n'%(epoch_i, tr_logloss, va_auc, va_logloss))
        torch.save(model, f'{save_dir}/{model_file_name}.pt')
    elif flag == 'new_train':
        assert omega is not None
        train_dataset = get_dataset(dataset_name, dataset_path, train_part, False, -1, '0')
        imp_train_dataset = get_dataset(dataset_name, dataset_path, train_part, False, -1, '1')
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, -1, '0')
        batch_ratio = len(imp_train_dataset) // len(train_dataset)
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=10, pin_memory=True, shuffle=True)
        imp_train_data_loader = DataLoader(imp_train_dataset, batch_size=int(batch_size*batch_ratio), num_workers=10, pin_memory=True, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)
        model = G(train_dataset.max_dim, embed_dim).to(device)
        if imp_type == 'r':
            stats = np.load('rnd_stats_%s.npy'%train_part)
            imp_model = stats.sum(axis=1)[0] / stats.sum(axis=1)[1]
            imp_model = np.log(imp_model/(1-imp_model))
            print('logit:', imp_model)
        elif imp_type == 'item-r':
            stats = np.load('rnd_stats_%s.npy'%train_part)
            r = stats.sum(axis=1)[0] / stats.sum(axis=1)[1]
            item_r = np.nan_to_num(stats[0, :] / stats[1, :], nan=r)
            item_r[item_r == 1] = r
            item_r[item_r < 1e-8] = r
            imp_model = np.log(item_r/(1-item_r))
            print('logit:', max(imp_model), min(imp_model))
            imp_model = torch.tensor(imp_model).to(device, torch.float)
        elif imp_type == 'complex':
            imp_model = torch.load('imp_%s.pt'%train_part, map_location=device)
            imp_model.eval()
        else:
            raise
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        model_file_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), \
                                                'bs-'+str(batch_size), 'k-'+str(embed_dim), 'o-'+str(omega), \
                                                train_part])
        with open(os.path.join(save_dir, model_file_name+'.log'), 'w') as log:
            for epoch_i in range(epoch):
                logloss, mseloss = new_train(model, imp_model, optimizer, train_data_loader, imp_train_data_loader, imp_type, device, omega)
                va_auc, va_logloss = test(model, valid_data_loader, device, model_name, 'wps')
                print('epoch:%d\ttr_logloss:%.6f\ttr_mseloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f'%(epoch_i, logloss, mseloss, va_auc, va_logloss))
                log.write('epoch:%d\ttr_logloss:%.6f\ttr_mseloss:%.6f\tva_auc:%.6f\tva_logloss:%.6f\n'%(epoch_i, logloss, mseloss, va_auc, va_logloss))
        torch.save(model, f'{save_dir}/{model_file_name}.pt')
    elif flag == 'gan_train':
        assert omega is not None
        full_dataset = get_dataset(dataset_name, dataset_path, 'select_'+train_part, False, -1, '1')  # total set
        det_dataset = get_dataset(dataset_name, dataset_path, 'det_'+train_part, False, -1, '0')  # S_c
        rnd_dataset = get_dataset(dataset_name, dataset_path, 'random_'+train_part, False, -1, '0')  # S_t
        full_data_loader = DataLoader(full_dataset, batch_size=batch_size*100, num_workers=8, pin_memory=True, shuffle=True)
        det_data_loader = DataLoader(det_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)
        rnd_data_loader = DataLoader(rnd_dataset, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)

        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, full_dataset.get_max_dim()-1, '0')
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=10, pin_memory=True)

        gen = G(full_dataset.max_dim, embed_dim).to(device)
        #dis = D(full_dataset.max_dim, embed_dim).to(device)
        dis = torch.load('./dis_trva.pt').to(device)

        opt_G = torch.optim.Adam(params=gen.parameters(), lr=learning_rate, weight_decay=weight_decay, )#amsgrad=True)
        opt_D = torch.optim.Adam(params=dis.parameters(), lr=learning_rate, weight_decay=weight_decay, )#amsgrad=True)
        model_file_name = '_'.join([model_name, 'lr-'+str(learning_rate), 'l2-'+str(weight_decay), \
                                                'bs-'+str(batch_size), 'k-'+str(embed_dim), 'o-'+str(omega), \
                                                train_part])
        with open(os.path.join(save_dir, model_file_name+'.log'), 'w') as log:
            for epoch_i in range(epoch):
                g_loss1, g_loss2, d_loss = gan_train(gen, dis, opt_G, opt_D, rnd_data_loader, det_data_loader, full_data_loader, device, omega, 0)
                va_auc, va_logloss = test(gen, valid_data_loader, device, model_name, 'wps')
                print('epoch:%d\tg_sup_loss:%.6f\tg_gan_loss:%.6f\td_loss:%.6f\tva_auc:%.6f\tva_logloss:%.6f'%(epoch_i, g_loss1, g_loss2, d_loss, va_auc, va_logloss))
                log.write('epoch:%d\tg_sup_loss:%.6f\tg_gan_loss:%.6f\td_loss:%.6f\tva_auc:%.6f\tva_logloss:%.6f\n'%(epoch_i, g_loss1, g_loss2, d_loss, va_auc, va_logloss))
        torch.save(gen, f'{save_dir}/{model_file_name}.pt')
    elif flag == 'pred':
        train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, train_dataset.get_max_dim() - 1, True)
        item_num = valid_dataset.get_item_num()
        refine_batch_size = int(batch_size//item_num*item_num)  # batch_size should be a multiple of item_num 
        valid_data_loader = DataLoader(valid_dataset, batch_size=refine_batch_size, num_workers=8, pin_memory=True)
        model = torch.load(model_path).to(device)
        pred(model, valid_data_loader, device, model_name, item_num)
    elif flag == 'test_auc':
        #train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, - 1)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
        #print(device)
        model = torch.load(model_path, map_location=device)
        va_auc, va_logloss = test(model, valid_data_loader, device, model_name, ps)
        print("model logloss auc")
        print("%s %.6f %.6f"%(model_name, va_logloss, va_auc))
        #pred(model, valid_data_loader, device, model_name, item_num)
    elif flag == 'obs_test_auc':
        #train_dataset = get_dataset(dataset_name, dataset_path, train_part, False)
        valid_dataset = get_dataset(dataset_name, dataset_path, valid_part, False, - 1, '2')
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=8, pin_memory=True)
        #print(device)
        model = torch.load(model_path, map_location=device)
        va_auc, va_logloss = obs_test(model, valid_data_loader, device, 'wps')
        print("model logloss auc")
        print("%s %.6f %.6f"%(model_name, va_logloss, va_auc))
        #pred(model, valid_data_loader, device, model_name, item_num)
    else:
        raise ValueError('Flag should be "train"/"pred"/"test_auc"!')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='yh')
    parser.add_argument('--train_part', default='tr')
    parser.add_argument('--valid_part', default='va')
    parser.add_argument('--dataset_path', help='the path that contains item.svm, va.svm, tr.svm trva.svm')
    parser.add_argument('--flag', default='train')
    parser.add_argument('--model_name', default='dssm')
    parser.add_argument('--model_path', default='', help='the path of model file')
    parser.add_argument('--epoch', type=float, default=30.)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=float, default=8192.)
    parser.add_argument('--embed_dim', type=float, default=16.)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--device', default='cuda:0', help='format like "cuda:0" or "cpu"')
    parser.add_argument('--omega', type=float, default=None)
    parser.add_argument('--save_dir', default='logs')
    parser.add_argument('--ps', default='wps')
    parser.add_argument('--imp_type', default=None)
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
         args.device,
         args.omega,
         args.save_dir,
         args.imp_type,
         args.ps)

