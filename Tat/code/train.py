import os,sys
import numpy as np
import math
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import esm
import copy
import random


from Rep2Mut import Rep2Mut

import argparse

import logging

from argparse import RawTextHelpFormatter
from scripts import read_signle_mut, getDictData


class train_model:
    def __init__(self,
                 seq_file, 
                 data, 
                 seq_n="Tat",
                 epoch = 200,
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 
                 learning_rate_1 = 1e-3, 
                 learning_rate_2 = 1e-5, 
                 batch_size = 8, 
                 save_path = "."
                ):
        self.epoch = epoch
        self.device = device
        self.seq_n = seq_n
        self.learning_rate_1 = learning_rate_1
        self.learning_rate_2 = learning_rate_2
        self.batch_size = batch_size
        self.save_path = save_path
        self.seq = seq_file
        self.data = data
        self.model_90_1, alphabet_90_1 = esm.pretrained.esm1v_t33_650M_UR90S_1()
        self.batch_converter_alphabet_90_1 = alphabet_90_1.get_batch_converter()
    def get_esm_vectors(self):
        print(self.seq)
        print(self.seq_n)
        wt_data = [(0, self.seq[self.seq_n])]
        _, _, batch_tokens = self.batch_converter_alphabet_90_1(wt_data)
        wt_results = self.model_90_1(batch_tokens, repr_layers=[33])
        mut_dict_keys = sorted(list(self.data.keys()))
        single_mut_act_list = []
        mut_seq_list = []
        mut_pos_list = []
        max_pos = 0;
        testi=0;
        for _mdk in mut_dict_keys:
            if self.data[_mdk].wt != self.seq[self.seq_n][ self.data[_mdk].pos ]:
                raise Exception("Error: not same: {} {}".format( self.seq[self.seq_n][ self.data[_mdk].pos ], self.data[_mdk] ))
            if self.data[_mdk].mut in ['_']: continue;
            single_mut_act_list.append( torch.FloatTensor( [self.data[_mdk].avg_activity/100] ))
            mut_seq_list.append( self.seq[self.seq_n][:self.data[_mdk].pos] +self.data[_mdk].mut + self.seq[self.seq_n][(self.data[_mdk].pos+1):])
            mut_pos_list.append( self.data[_mdk].pos )
            if self.data[_mdk].pos > max_pos:
                max_pos = self.data[_mdk].pos
        print("Generate mean represetations for seqs")
        sys.stdout.flush()
        mut_data = [(0, self.seq[self.seq_n])]
        _, _, batch_tokens = self.batch_converter_alphabet_90_1(mut_data)
        wt_rep = self.model_90_1(batch_tokens, repr_layers=[33])
        wt_rep = torch.mean(wt_rep['representations'][33], 1)[0].detach()
        seq_repr_mt_list = []
        seq_repr_wt_list = []
        mut_det = []
        max_pos += 1;
        pos_temp_v = torch.Tensor([0 for _ in range(max_pos)])
        pos_list = []
        for _iseq_ind in range(len(mut_seq_list)):
            _iseq = mut_seq_list[_iseq_ind]
            mut_data = [(0, _iseq)]
            _, _, batch_tokens = self.batch_converter_alphabet_90_1(mut_data)
            results = self.model_90_1(batch_tokens, repr_layers=[33])
            seq_repr_wt_list.append( wt_results['representations'][33][0][ mut_pos_list[_iseq_ind]+1 ].detach() )
            seq_repr_mt_list.append(    results['representations'][33][0][ mut_pos_list[_iseq_ind]+1 ].detach() )
            pos_list.append(         copy.deepcopy(pos_temp_v))
            mut_det.append(self.seq[self.seq_n][mut_pos_list[_iseq_ind]]+str(mut_pos_list[_iseq_ind])+_iseq[mut_pos_list[_iseq_ind]])
            pos_list[-1][mut_pos_list[_iseq_ind]] = 1;
        self.train_data = (seq_repr_wt_list, seq_repr_mt_list, single_mut_act_list, pos_list, mut_det)
    def load_model(self, f = "/mnt/analysis/derbelh/ProtMut_prediction_private-master/General_prediction/code/product/product_2.p"):
        self.model = Rep2Mut(1280, [self.seq_n])
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(f), strict=False)
    def train(self):
        random_seed = 4
        g_adam_betas=(0.9,0.999)
        g_adam_weight_decay=0.01
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        target_list = []
        pred_list = []
        add_firstn = 0;
        data_train = {}
        loss_func  = torch.nn.MSELoss();
        adam_optmizer = torch.optim.Adam(
        [
        {"params": self.model.wt_linear1.parameters(), "lr": self.learning_rate_2},
        {"params": self.model.mt_linear1.parameters(), "lr": self.learning_rate_2},
        {"params": self.model.act1.parameters(), "lr": self.learning_rate_2},
        {"params": self.model.act2.parameters(), "lr": self.learning_rate_2},
        {"params": self.model.outputs.parameters(), "lr": self.learning_rate_1},
        ],lr=1e-4, betas=g_adam_betas, weight_decay=g_adam_weight_decay)
        epochi = 0;
        num_train_i = 0
        while epochi < self.epoch:
            self.model.train()
            losses = 0
            x = 0
            train_loader = getDictData({self.seq_n: self.train_data}.copy(), True, self.batch_size)
            for train_batch in train_loader:
                num_train_i += 1
                x += 1
                pred_mask = self.model.forward(torch.stack(train_batch[1][0]).to(self.device, dtype=torch.float64), torch.stack(train_batch[1][1]).to(self.device, dtype=torch.float64), torch.stack(train_batch[1][3]).to(self.device, dtype=torch.float64), train_batch[0])
                loss = loss_func(pred_mask.to(self.device), torch.stack(train_batch[1][2]).to(self.device, dtype=torch.float64) )
                adam_optmizer.zero_grad();
                loss.backward();
                adam_optmizer.step();
                losses += loss
            epochi += 1
            print(losses/x)
            self.model.eval()
    def save(self):
        torch.save(self.model.state_dict(), self.save_path)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Train Rep2Mut.", formatter_class=argparse.ArgumentDefaultsHelpFormatter);

    parser.add_argument("-s", type=str, 
                                     help="The file containing the protein sequence. ");
    parser.add_argument("-n", type=str, 
                                     help="The name of the protein sequence in the file. ");
    parser.add_argument("-m", type=str, 
                                     help="The file containing the mutations, the position of the mutations and the output values.");
    parser.add_argument("-a", type=str, default= ' Percentage of Reads GFP High over High and Low', 
                                     help="Column name of the activity. ");
    parser.add_argument("-p", type=str, default=' Position', 
                                     help="Column name of the position. ");
    parser.add_argument("-v", type=str, default='#Variant ID', 
                                     help="Column name of the Variant ID.");
    parser.add_argument("-bt", type=int, default=5, 
                                     help="Barcode Count threshold. ");
    parser.add_argument("-bc", type=str, 
                                     help="Column name of the Barcode Count.");
    parser.add_argument("-epoch", type=int, default=200, 
                                     help="The number of epochs. ");
    parser.add_argument("-gpu", type=str, default="0",
                                     help="The device to use to train the model. ");
    parser.add_argument("-lr1", type=float, default=1e-3,
                                     help="The learning rate of the task specific layer.");
    parser.add_argument("-lr2", type=float, default=1e-5,
                                     help="The learning rate of the shared task. ");
    parser.add_argument("-batch", type=int, default=8, 
                                     help="The batch size. ");
    parser.add_argument("-save", type=str, default="./Rep2Mut.p", 
                                     help="The save folder. ");
  
    args = parser.parse_args()
        
        
    mut = read_signle_mut(args.m, args.n, args.s, args.a, args.p, args.v, args.bt, args.bc)
    mut_dict = mut.read_signle_mut()
    
    
    tatseq = mut.seq
    model = train_model(tatseq, 
                        mut_dict,
                        args.n,
                        args.epoch,
                        torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu"),
                        args.lr1,
                        args.lr2,
                        args.batch,
                        args.save,
                       )
    model.get_esm_vectors()
    model.load_model()
    model.train()
    model.save()
