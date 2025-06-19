import os
import sys
import random

import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

import util


def load_data(data_name):
    main_dir = sys.path[0]
    x_list = []
    y_list = []
    mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))

    if data_name in ['HandWritten']:
        x_all = mat['X'][0]
        for view in range(x_all.shape[0]):
            x_list.append(x_all[view])
        y = np.squeeze(mat['Y']).astype('int')
        y_list.append(y)

    elif data_name in ['Caltech101-7']:
        x_all = mat['X'][0]
        for view in range(x_all.shape[0]):  # [3, 4]:
            x_list.append(x_all[view])
        y = np.squeeze(mat['Y']).astype('int')
        y_list.append(y)

    elif data_name in ['aloideep3v']:
        x_all = mat['X'][0]
        for view in range(x_all.shape[0]):
            x_list.append(x_all[view])
        y = np.squeeze(mat['truth']).astype('int')
        y_list.append(y)

    elif data_name in ['Scene-15']:
        x_all = mat['X'][0]
        for view in range(x_all.shape[0]):
            x_list.append(x_all[view])
        y_list.append(np.squeeze(mat['Y']))

    elif data_name in ['Fashion']:
        x_list.append(mat['X1'].reshape(10000, -1))
        x_list.append(mat['X2'].reshape(10000, -1))
        x_list.append(mat['X3'].reshape(10000, -1))
        y_list.append(np.squeeze(mat['Y']))
    
    elif data_name in ['CIFAR10']:
        x_all = mat['new_data']
        x_list.append(x_all[0][0].T)
        x_list.append(x_all[1][0].T)
        x_list.append(x_all[2][0].T)
        y_list.append(np.squeeze(mat['new_truelabel'][0][0]))
        
    # switch y
    y = y_list[0]
    y = y - np.min(y)
    y_list[0] = y

    return x_list, y_list

def load_data_recover(data_name):
    main_dir = sys.path[0]
    x_list = []
    y_list = []
    mat = sio.loadmat(os.path.join(main_dir, 'data/recover_data', data_name + '_recover.mat'))
    if data_name in ['HandWritten']:
        x_list.append(mat['X1'])
        x_list.append(mat['X2'])
        x_list.append(mat['X3'])
        x_list.append(mat['X4'])
        x_list.append(mat['X5'])
        y = np.squeeze(mat['Y']).astype('int')
        y_list.append(y)
    
    if data_name in ['Fashion']:
        x_list.append(mat['X1'])
        x_list.append(mat['X2'])
        x_list.append(mat['X3'])
        y_list.append(np.squeeze(mat['Y']))
        
    if data_name in ['Scene-15']:
        x_list.append(mat['X1'])
        x_list.append(mat['X2'])
        x_list.append(mat['X3'])
        y_list.append(np.squeeze(mat['Y']))
        
    if data_name in ['aloideep3v']:
        x_list.append(mat['X1'])
        x_list.append(mat['X2'])
        x_list.append(mat['X3'])
        y_list.append(np.squeeze(mat['Y']))
    
    if data_name in ['Caltech101-7']:
        x_list.append(mat['X1'])
        x_list.append(mat['X2'])
        x_list.append(mat['X3'])
        x_list.append(mat['X4'])
        x_list.append(mat['X5'])
        x_list.append(mat['X6'])
        y_list.append(np.squeeze(mat['Y']))
        
    if data_name in ['CIFAR10']:
        x_list.append(mat['X1'])
        x_list.append(mat['X2'])
        x_list.append(mat['X3'])
        y_list.append(np.squeeze(mat['Y']))
        
    return x_list, y_list

def get_incomplete_idx(data_name, missing_rate):
    main_dir = sys.path[0]
    mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '_percentDel_' + str(missing_rate) + '.mat'))
    folds_data = mat['folds']
    inc_idx = np.array(folds_data[0, 0], 'int32')  # shape is [n x v]

    random.seed(1)
    total_sample_num = inc_idx.shape[0]
    sample_index = list(range(total_sample_num))
    random.shuffle(sample_index)
    sample_index = np.array(sample_index)

    return inc_idx[sample_index]


def norm_data(data_name, x_list):
    
    ss_list = None
    if data_name in ['HandWritten','aloideep3v', 'Caltech101-7','Scene-15', 'Fashion', 'CIFAR10']:
        ss_list = [StandardScaler() for _ in range(len(x_list))]
        x_list_new = [ss_list[v].fit_transform(v_data.astype(np.float32)) for v, v_data in enumerate(x_list)]
    # elif data_name in ['BDGP']:
    #     x_list_new = [util.normalize(x).astype('float32') for x in x_list]
    # elif data_name in ['BDGP']:
    #     x_list_new = [util.normalize_row(x).astype('float32') for x in x_list]
    else:
        x_list_new = [x.astype('float32') for x in x_list]
    return x_list_new,ss_list


class ComDataset(Dataset):
    def __init__(self, fea, inc, device):
        self.device = device
        self.fea = fea
        self.inc = inc

    def __getitem__(self, index):
        return [torch.from_numpy(x[index]).to(self.device) for x in self.fea], \
            torch.from_numpy(self.inc[index]).to(self.device), index

    def __len__(self):
        return self.fea[0].shape[0]


def get_loader(config, device):
    # load data from disk
    x, y = load_data(config['Dataset']['name'])
    y = y[0]

    # load incomplete idx
    inc_idx = get_incomplete_idx(config['Dataset']['name'], config['Dataset']['missing_rate'])

    # norm
    x,ss_list = norm_data(config['Dataset']['name'], x)

    # mask invalid data
    masked_x = []
    for i, view in enumerate(x):
        # 为当前视图的特征创建掩码，将mask扩展到特征维度
        view_mask = inc_idx[:, i][:, np.newaxis]
        # 将特征矩阵与掩码相乘，实现特征置零
        masked_view = view * view_mask
        masked_x.append(masked_view.astype('float32'))

    # construct loader
    dataset = ComDataset(masked_x, inc_idx, device)
    data_loader = DataLoader(
        dataset,
        batch_size=config['Dataset']['batch_size'],
        shuffle=True
    )
    
    return data_loader, x, y, inc_idx, masked_x, ss_list


def get_loader_recover(config,device):
    # load data from disk
    x, y = load_data_recover(config['Dataset']['name'])
    y = y[0]

    # load incomplete idx
    inc_idx = get_incomplete_idx(config['Dataset']['name'], config['Dataset']['missing_rate'])

    # norm
    x,ss_list = norm_data(config['Dataset']['name'], x)

    # mask invalid data
    masked_x = []
    for i, view in enumerate(x):
        # 为当前视图的特征创建掩码，将mask扩展到特征维度
        view_mask = inc_idx[:, i][:, np.newaxis]
        # 将特征矩阵与掩码相乘，实现特征置零
        masked_view = view * view_mask
        masked_x.append(masked_view.astype('float32'))

    # construct loader
    dataset = ComDataset(x, inc_idx, device)
    data_loader = DataLoader(
        dataset,
        batch_size=config['Dataset']['batch_size'],
        shuffle=True
    )
    
    return data_loader, x, y, inc_idx, masked_x, ss_list
