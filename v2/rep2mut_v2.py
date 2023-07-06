import os


import os,sys
import pandas as pd
import math
import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import torch
import esm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse


import scipy
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from scipy import stats


import csv
import pickle

WARNING_L = 3;
INFO_L = 2;
GEN_L = 1;

g_flanking_size = 400
g_input_dim = 1280
g_adam_learning_rate =1e-3
g_adam_betas=(0.9,0.999)
g_adam_weight_decay=0.01
g_epoch = 30;
g_batch_size = 64
g_random_seed = 3

output_level = WARNING_L






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
            l1, l2, l3, l4= zip(*temp)
            self.data = [list(l1), list(l2), list(l3), list(l4)]
        return self
    def __next__(self):
        if(self.batch_size == 0):
            self.n = len(self.data[0])
            return self.data[0],self.data[1],self.data[2],self.data[3]
        if self.n < len(self.data[0]):
            a = self.n
            self.n += self.batch_size
            return (self.data[0][a:self.n],self.data[1][a:self.n],self.data[2][a:self.n])
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





def Train_mut(act_net2, data_train, epoch, lr1, lr2, batch, save):
	epochi = 0;
	num_train_i = 0        
	loss_func2  = torch.nn.MSELoss();
	b_size = batch 
	adam_learning_rate2 = lr1
	adam_learning_rate1 = lr2
	adam_optmizer2 = torch.optim.Adam(
	[
	{"params": act_net2.wt_linear1.parameters(), "lr": adam_learning_rate2},
	{"params": act_net2.mt_linear1.parameters(), "lr": adam_learning_rate2},
	{"params": act_net2.act1.parameters(), "lr": adam_learning_rate2},
	{"params": act_net2.act2.parameters(), "lr": adam_learning_rate2},
	{"params": act_net2.outputs.parameters(), "lr": adam_learning_rate1}
	],lr=1e-4, betas=g_adam_betas, weight_decay=g_adam_weight_decay)
	epoch1 = epoch
	epochi1 = 0
	while epochi1 <= epoch1:
		act_net2.train()
		num_train_i = 0
		test_loader1 = getDictData(data_train.copy(), True, batch)
		losses2 = 0
		for train_batch1 in test_loader1:
			num_train_i += 1
			pred_mask = act_net2.forward(torch.stack(train_batch1[1][0]).to(device, dtype=torch.float64), torch.stack(train_batch1[1][1]).to(device, dtype=torch.float64), train_batch1[0])
			loss = loss_func2(pred_mask.to(device), torch.stack(train_batch1[1][2]).to(device, dtype=torch.float64) )
			adam_optmizer2.zero_grad()
			loss.backward()
			adam_optmizer2.step()
			losses2 += loss
		print("Training loss = ", losses2/num_train_i)
		epochi1 += 1
	torch.save(act_net2.state_dict(), save)


            
def predict(_data, model_Rep2Mut, csv_file, device, d):
	_loader = getDictData({d : _data}, False, 100000)
	for e in _loader:
		print(d)
		pred_mask = model_Rep2Mut.forward(torch.stack(e[1][0]).to(device, dtype=torch.float64), torch.stack(e[1][1]).to(device, dtype=torch.float64), d)
	pred_mask = torch.squeeze(pred_mask).tolist()
	mut_ = _data[3]
	d = {"mut":mut_, "Rep2Mut prediction":pred_mask}
	df = pd.DataFrame(d)
	df.to_csv(csv_file)
            
            
if __name__=='__main__':
  parser = argparse.ArgumentParser(description="Train Rep2Mut.", formatter_class=argparse.ArgumentDefaultsHelpFormatter);
  
  parser.add_argument("-d", type=str, default=None, 
                                     help="The name of the datasetset to train/test. ");
  parser.add_argument("-gpu", type=str, default="0",
                                     help="The device to use to train the model. ");
  parser.add_argument("-p", type=str, default="./vectors.p", 
                                     help="the file containing the vectors generated by generate_vectors.py")
  
  
  subparsers = parser.add_subparsers(dest='command')
  
  parser_train = subparsers.add_parser("train", help="Train the model")
  
  parser_train.add_argument("-epoch", type=int, default=200, 
                                     help="The number of epochs. ");
  parser_train.add_argument("-lr1", type=float, default=1e-3,
                                     help="The learning rate of the task specific layer.");
  parser_train.add_argument("-lr2", type=float, default=1e-5,
                                     help="The learning rate of the shared task. ");
  parser_train.add_argument("-batch", type=int, default=8, 
                                     help="The batch size. ");
  parser_train.add_argument("-save", type=str, default="./Rep2Mut.p", 
                                     help="The save folder. ");
  
  parser_test = subparsers.add_parser("test", help="Test the model")
  
  parser_test.add_argument("-m", type=str, default="./Rep2Mut.p", 
                                     help="The model parameters. ");
  
  parser_test.add_argument("-save", type=str, default="./output.csv", 
                                     help="The save folder. ");
    
  args = parser.parse_args()
  
  device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
  
  vectors = pickle.load(open("./vectors.p","rb"))
  if args.command  == 'train':
    if args.d == None: datasets = vectors
    else: datasets = {args.d : vectors[args.d]}

    act_net2 = Rep2Mut(g_input_dim, datasets);
    act_net2 = act_net2.to(device)

    Train_mut(act_net2, datasets.copy(), args.epoch, args.lr1, args.lr2,args.batch, args.save)
    
  elif args.command == 'test':
    act_net2 = Rep2Mut(g_input_dim, [args.d]);
    act_net2.load_state_dict(torch.load(args.m)) 
    act_net2.eval()
    act_net2 = act_net2.to(device)
    predict(vectors[args.d], act_net2, args.save, device, args.d)
