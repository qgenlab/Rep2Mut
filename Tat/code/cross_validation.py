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

from predict import get_res
from train import train_model

from scripts import read_signle_mut, getDictData, read_signle_mut_predict


from argparse import RawTextHelpFormatter

    

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Cross validation.", formatter_class=argparse.ArgumentDefaultsHelpFormatter);
    parser.add_argument("-save", type=str, default="./0.p", 
                                     help="The output file. ");
    parser.add_argument("-s", type=str, 
                                     help="The file containing the protein sequence. ");
    parser.add_argument("-n", type=str, 
                                     help="The name of the protein sequence in the file. ");
    parser.add_argument("-m", type=str, 
                                     help="The file containing the mutations, the position of the mutations and the output values.");
    parser.add_argument("-a", type=str, default= ' Percentage of Reads GFP High over High and Low', 
                                     help="Row name of the activity. ");
    parser.add_argument("-p", type=str, default=' Position', 
                                     help="Row name of the position. ");
    parser.add_argument("-v", type=str, default='#Variant ID', 
                                     help="Row name of the Variant ID.");
    parser.add_argument("-bt", type=int, default=5, 
                                     help="Barcode Count threshold. ");
    parser.add_argument("-bc", type=str, 
                                     help="Row name of the Barcode Count.");
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
    parser.add_argument("-f", type=int, default=10, 
                                     help="The number of folds. ");
    
    args = parser.parse_args()
        
    n_folds = args.f
    
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
    
    dict_ = model.train_data
    mut_p = read_signle_mut_predict(args.m, args.n, args.s, args.p, args.v)
    mut_dict_p = mut_p.read_signle_mut()        
    
    
    model_p = get_res(args.save, args.s, mut_dict,args.n,torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu"),args.save)
    
    df = pd.DataFrame()
    for e in range(n_folds):
        model.load_model()
        x = int(e*len(dict_[0])/n_folds)
        y = int(e*len(dict_[0])/n_folds +len(dict_[0])/n_folds)
        model.train_data = (dict_[0][:x] + dict_[0][y:] , dict_[1][:x] + dict_[1][y:], dict_[2][:x] + dict_[2][y:], dict_[3][:x] + dict_[3][y:], dict_[4][:x] + dict_[4][y:])
        model_p._data = (dict_[0][x:y], dict_[1][x:y], dict_[3][x:y], dict_[4][x:y])
        model.train()
        model.save()
        model_p.load_model()
        model_p.predict()
        df = pd.concat([df, model_p.df], ignore_index=True)
        print(df)
    df.to_csv("Cross_validation.csv", index=False)

    
    
    

