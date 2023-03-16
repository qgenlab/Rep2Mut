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
from scripts import read_signle_mut_predict, getDictData


class get_res:
    def __init__(self,
                 f,
                 model,
                 seq_file, 
                 data, 
                 seq_n="Tat",
                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),  
                 save_path = "./0.csv"
                ):
        self.file = pd.read_csv(f)
        self.model = model
        self.model_Rep2Mut = None
        self.device = device
        self.seq_n = seq_n
        self.save_path = save_path
        self.seq = seq_file
        self.data = data
        self._data = None
        self.model_90_1, alphabet_90_1 = esm.pretrained.esm1v_t33_650M_UR90S_1()
        self.batch_converter_alphabet_90_1 = alphabet_90_1.get_batch_converter()
    def get_esm_vectors(self):
        print(self.seq)
        print(self.seq_n)
        wt_data = [(0, self.seq[self.seq_n])]
        _, _, batch_tokens = self.batch_converter_alphabet_90_1(wt_data)
        wt_results = self.model_90_1(batch_tokens, repr_layers=[33])
        mut_dict_keys = sorted(list(self.data.keys()))
        mut_seq_list = []
        mut_pos_list = []
        max_pos = 0;
        testi=0;
        for _mdk in mut_dict_keys:
            if mut_dict[_mdk].wt != self.seq[self.seq_n][ mut_dict[_mdk].pos ]:
                raise Exception("Error: not same: {} {}".format( self.seq[self.seq_n][ mut_dict[_mdk].pos ], mut_dict[_mdk] ))
            if mut_dict[_mdk].mut in ['_']: continue;
            mut_seq_list.append( self.seq[self.seq_n][:mut_dict[_mdk].pos] +mut_dict[_mdk].mut + self.seq[self.seq_n][(mut_dict[_mdk].pos+1):])
            mut_pos_list.append( mut_dict[_mdk].pos )
            if mut_dict[_mdk].pos > max_pos:
                max_pos = mut_dict[_mdk].pos
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
        self._data = (seq_repr_wt_list, seq_repr_mt_list, pos_list, mut_det)
    def load_model(self):
        self.model_Rep2Mut = Rep2Mut(1280, [self.seq_n])
        self.model_Rep2Mut = self.model_Rep2Mut.to(self.device)
        self.model_Rep2Mut.load_state_dict(torch.load(self.model), strict=False)
    def predict(self):
        _loader = getDictData({self.seq_n: self._data}.copy(), False, 100000)
        for e in _loader:
            pred_mask = self.model_Rep2Mut.forward(torch.stack(e[1][0]).to(self.device, dtype=torch.float64), torch.stack(e[1][1]).to(self.device, dtype=torch.float64), torch.stack(e[1][2]).to(self.device, dtype=torch.float64), e[0])
        pred_mask = torch.squeeze(pred_mask).tolist()
        mut_ = self._data[3]
        for i, e in enumerate(mut_):
            mut_[i] = e[0]+str(int(e[1:-1])+1)+e[-1]
        d = {"mut":mut_, "Rep2Mut prediction":pred_mask}
        self.df = pd.DataFrame(d)
    def save(self):
        print(self.file)
        print(self.df)
        self.df_join = pd.merge(self.file, self.df, how='outer', left_on = '#Variant ID'  , right_on ='mut')
        print(self.df_join)
        self.df_join.drop(columns=["mut"]).to_csv(self.save_path, index=False)
        
        
        
if __name__=='__main__':       
    parser = argparse.ArgumentParser(description="Determine the GigaAssay of a mutation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter);

    parser.add_argument("-s", type=str, 
                                     help="The file containing the protein sequence. ")
    parser.add_argument("-model", type=str, default='/mnt/analysis/derbelh/ProtMut_prediction_private-master/General_prediction/code/product/product_2.p',
                                     help="The model. ");
    parser.add_argument("-n", type=str, 
                                     help="The name of the protein sequence in the file. ");
    parser.add_argument("-m", type=str, 
                                     help="The file containing the mutations, the position of the mutations.");
    parser.add_argument("-p", type=str, default=' Position', 
                                     help="Row name of the position. ");
    parser.add_argument("-v", type=str, default='#Variant ID', 
                                     help="Row name of the Variant ID.");
    parser.add_argument("-gpu", type=str, default="0",
                                     help="The device to use to train the model. ");
    parser.add_argument("-save", type=str, default="./0.csv", 
                                     help="The output file. ");
    args = parser.parse_args()
    
    
    
    mut = read_signle_mut_predict(args.m, args.n, args.s, args.p, args.v)
    mut_dict = mut.read_signle_mut()        
    
    tatseq = mut.seq
    
    model = get_res(args.m, args.model, tatseq, mut_dict,args.n,torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu"),args.save)
    #model = get_res(args.model, args.s, mut_dict,args.n,torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu"),args.save)
    model.load_model()
    model.get_esm_vectors()
    model.predict()
    model.save()
