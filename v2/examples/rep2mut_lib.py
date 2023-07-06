import os

import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt


import os,sys
import pandas as pd
import math
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import scipy
from scipy import stats

import pickle

import math



WARNING_L = 3;
INFO_L = 2;
GEN_L = 1;

g_flanking_size = 400
g_input_dim = 1280
g_adam_learning_rate =1e-3
g_adam_betas=(0.9,0.999)
g_adam_weight_decay=0.01
g_epoch = 30;
g_random_seed = 3

output_level = WARNING_L
import csv

perf = {}



class Rep2Mut(nn.Module):
    def __init__(self, input_dim, outputs = ['0'], dropout_rate=0.2):
        super(Rep2Mut, self).__init__();
        self.input_dim = input_dim; #1280
        self.hidden_dim1 = 128;
        self.pos_dim = 86;
        self.hidden_dim2 = 1;
        self.wt_linear1 = nn.Linear(self.input_dim, self.hidden_dim1, dtype=torch.float64)
        self.mt_linear1 = nn.Linear(self.input_dim, self.hidden_dim1, dtype=torch.float64)
        self.wt_dropout_lary = nn.Dropout(dropout_rate)
        self.mt_dropout_lary = nn.Dropout(dropout_rate)
        self.act1 = nn.PReLU(dtype=torch.float64)
        self.act2 = nn.PReLU(dtype=torch.float64)
        self.outputs = nn.ModuleDict({})
        for e in outputs:
            self.outputs[e] = nn.Linear(self.hidden_dim1, self.hidden_dim2, dtype=torch.float64)
    def forward(self, wt_x, mt_x, output = '0'):
        wt_x = self.act1(self.wt_linear1(wt_x))
        wt_x = self.wt_dropout_lary(wt_x)
        mt_x = self.act2(self.mt_linear1(mt_x))
        mt_x = self.mt_dropout_lary(mt_x)
        if output=="Tat":a_x = torch.sigmoid(self.outputs[output]( wt_x   *  mt_x));
        else: a_x = self.outputs[output]( wt_x   *  mt_x);
        return a_x



class getData:
    def __init__(self, data, shuffle, batch_size=0):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max = len(self.data[0]) - 1
    def __iter__(self):
        self.n = 0
        self.data_n = 0
        if self.shuffle == True:
            temp = list(zip(self.data[0],self.data[1],self.data[2],self.data[3]))
            random.shuffle(temp)
            l1, l2, l3, l4 = zip(*temp)
            self.data = [list(l1), list(l2), list(l3), list(l4)]
        return self
    def __next__(self):
        if(self.batch_size == 0):
            self.n = len(self.data[0])
            return self.data[0],self.data[1],self.data[2],self.data[3]
        if self.n < len(self.data[0]):
            a = self.n
            self.n += self.batch_size
            return (self.data[0][a:self.n],self.data[1][a:self.n],self.data[2][a:self.n],self.data[3][a:self.n])
        else:
            raise StopIteration


class getDictData:
    def __init__(self, data, shuffle, batch_size):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        for e in self.data:
            element = iter(getData(self.data[e], self.shuffle, self.batch_size))
            self.data[e] = element
    def __iter__(self):
        self.keys = list(self.data.keys())
        if self.shuffle == True:
            random.shuffle(self.keys)
        return self
    def __next__(self):
        batch = 0
        while batch == 0:
            if self.keys == []:
                raise StopIteration
            key = random.choice(self.keys)
            try:
                batch = next(self.data[key])
            except StopIteration:
                self.keys.remove(key)
        return key, batch


def GetCorrelation(d1, d2):
    perf = {}
    pearson = scipy.stats.pearsonr(d1, d2)[0]
    spearman = scipy.stats.spearmanr(d1, d2)[0]
    perf = {'Pearson':pearson,'Spearman':spearman}
    return perf[e]



def cross_validation(data_test1, device, _x0 ):
    n_folds = 10
    out = []
    re = []
    rr1 = []
    rr2 = []
    mutations = []
    temp = list(zip(data_test1[_x0][0], data_test1[_x0][1], data_test1[_x0][2], data_test1[_x0][3]))
    random.shuffle(temp)
    l1, l2, l3, l4 = zip(*temp)
    data_test1[_x0] = (list(l1), list(l2), list(l3), list(l4))
    for _i in range(n_folds):
        _x = int(_i *len(data_test1[_x0][0])/n_folds)
        _y = int(_i*len(data_test1[_x0][0])/n_folds +len(data_test1[_x0][0])/n_folds)
        print("training the model: ", _i)
        act_net2 = Rep2Mut(g_input_dim, [_x0]);
        act_net2 = act_net2.to(device)
        loss_func2  = torch.nn.MSELoss();
        b_size = 16   
        adam_learning_rate2 = 5e-6 
        adam_learning_rate1 = 1e-4
        adam_optmizer2 = torch.optim.Adam(
        [
        {"params": act_net2.wt_linear1.parameters(), "lr": adam_learning_rate2},
        {"params": act_net2.mt_linear1.parameters(), "lr": adam_learning_rate2},
        {"params": act_net2.act1.parameters(), "lr": adam_learning_rate2},
        {"params": act_net2.act2.parameters(), "lr": adam_learning_rate2},
        {"params": act_net2.outputs.parameters(), "lr": adam_learning_rate1},
        ],lr=1e-5, betas=g_adam_betas, weight_decay=g_adam_weight_decay)
        epoch1 = 200 
        epochi1 = 0
        data_test = data_test1[_x0]
        train_ = (data_test[0][:_x] + data_test[0][_y:] , data_test[1][:_x] + data_test[1][_y:], data_test[2][:_x] + data_test[2][_y:], data_test[3][:_x] + data_test[3][_y:])
        test_ = (data_test[0][_x:_y], data_test[1][_x:_y], data_test[2][_x:_y], data_test[3][_x:_y])
        while epochi1 <= epoch1:
            act_net2.train()
            num_train_i = 0
            test_loader1 = getDictData({_x0 : train_}, True, b_size)
            losses2 = 0
            x =  0
            for train_batch1 in test_loader1:
                num_train_i += 1
                x += 1
                pred_mask = act_net2.forward(torch.stack(train_batch1[1][0]).to(device, dtype=torch.float64), torch.stack(train_batch1[1][1]).to(device, dtype=torch.float64), train_batch1[0])
                loss = loss = loss_func2(pred_mask.to(device), torch.stack(train_batch1[1][2]).to(device, dtype=torch.float64) )
                adam_optmizer2.zero_grad();
                loss.backward();
                adam_optmizer2.step();
                losses2 += loss
            epochi1 += 1
            out = []
            test_loader2 = getDictData({_x0 : test_}, False, 20000)
            loss_test = 0
            d_test = {}
            perf = {}
            _out =  []
        for e1 in test_loader2:
            pred_mask = act_net2.forward( torch.stack(e1[1][0], 0).to(device, dtype=torch.float64), torch.stack(e1[1][1], 0).to(device, dtype=torch.float64),  e1[0])
            loss = loss_func2(pred_mask, torch.stack(e1[1][2]).to(device, dtype=torch.float64) )
            loss_test += loss
            try:
               _out = (torch.cat((d_test[e1][0], pred_mask)), torch.cat(( d_test[e1][1], torch.stack(e1[1][2]).to(device, dtype=torch.float64))), e1[1][3])
            except TypeError:
               _out = (pred_mask, torch.stack(e1[1][2]).to(device, dtype=torch.float64), e1[1][3])
        __out = [_out[0].squeeze().cpu().detach().numpy(), _out[1].squeeze().cpu().detach().numpy(), _out[2]]
        print("This {} loss: {} for ep {}".format(_i,  losses2/num_train_i, num_train_i))
        rr1.extend(__out[0].tolist())
        rr2.extend(__out[1].tolist())
        mutations.extend(__out[2])
    return rr1, rr2, mutations



