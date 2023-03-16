import os,sys
import numpy as np
import pandas as pd
import math


import copy
import random

import re

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
            temp = list(zip(self.data[0],self.data[1],self.data[2],self.data[3], self.data[4]))
            random.shuffle(temp)
            l1, l2, l3, l4, l5 = zip(*temp)
            self.data = [list(l1), list(l2), list(l3), list(l4), list(l5)]
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


    
    
class read_signle_mut_predict:
    def __init__(self, File, seq_n, seq, pos_str, variant_str):
        self._single_mutation = pd.read_csv(File)
        self.seq_n = seq_n
        self.seq = self.read_fa(seq, seq_n)
        self.pos_str = pos_str
        self.variant_str = variant_str
        print(File)
    def read_fa(self, fafile, seq_n):
        seq_dict = {}
        prev_id = ''
        prev_list = []
        with open(fafile, 'r') as mr:
            while True:
                line = mr.readline();
                if not line: break;
                line = line.strip();
                if len(line)==0: continue;

                if line[0] in ['>'] and line[1:] == seq_n:
                    if len(prev_list)>0:
                        seq_dict[ prev_id ] = ''.join( prev_list );
                        prev_id  = ''
                        prev_list = []
                    prev_id = line[1:].split()[0];
                else:
                    prev_list.append( line );
            if len(prev_list)>0:
                seq_dict[ prev_id ] = ''.join( prev_list );
        return seq_dict
    def read_signle_mut(self):
        mut_dict = {}
        for rowind in range(self._single_mutation.shape[0]):
            single_row = self._single_mutation.iloc[rowind]
            t_pos = single_row[self.pos_str]
            t = single_row[self.variant_str]
            mutinfo = t.split(str(t_pos))
            wt = mutinfo[0]
            if wt==mutinfo[1]: continue;
            t_pos = t_pos - 1
            if wt[0]==self.seq[self.seq_n][t_pos]: pass
            else: print("Error: {} {} {}".format( single_row[self.variant_str], t_pos, self.seq[wt[0]][t_pos] ))
            mkey = "{}{}{}".format( wt, t_pos, mutinfo[1].strip() )
            if mkey not in mut_dict and re.match(r'(?i)[rhkdestnqcugpavilmfyw]\d*[rhkdestnqcugpavilmfyw]', mkey):
                mut_dict[mkey] = single_mut(wt, t_pos, mutinfo[1])
        return mut_dict

class read_signle_mut(read_signle_mut_predict):
    def __init__(self, File, seq_n, seq, activity_str, pos_str, variant_str, barcode_thr, barcode_column_str):
        super().__init__(File, seq_n, seq, pos_str, variant_str);
        self.activity_str = activity_str
        self.barcode_column_str = barcode_column_str
        self.barcode_thr = barcode_thr
    def read_signle_mut(self):
        mut_dict = {}
        for rowind in range(self._single_mutation.shape[0]):
            single_row = self._single_mutation.iloc[rowind]
            t_pos = single_row[self.pos_str]
            t = single_row[self.variant_str]
            mutinfo = t.split(str(t_pos))
            wt = mutinfo[0]
            if wt==mutinfo[1]: continue;
            t_pos = t_pos - 1
            if wt[0]==self.seq[self.seq_n][t_pos]: pass
            else: print("Error: {} {} {}".format( single_row[self.variant_str], t_pos, self.seq[wt[0]][t_pos] ))
            mkey = "{}{}{}".format( wt, t_pos, mutinfo[1].strip() )
            if mkey not in mut_dict:
                mut_dict[mkey] = single_mut(wt, t_pos, mutinfo[1])
            if(single_row[self.barcode_column_str]>=self.barcode_thr):
                mut_dict[mkey].add_actitivity(single_row[self.activity_str])
        del_keys = []
        for mk in mut_dict:
            if mut_dict[mk].avg_activity==-1:
                del_keys.append(mk)
                continue;
            if len(mut_dict[mk].activity_list)<3:
                print("{} {}".format( mk, mut_dict[mk]   ))
                del_keys.append(mk)
        for _dk in del_keys:
            del mut_dict[_dk]
        return mut_dict


class single_mut_pred:
    def __init__(self, wt, pos, mut):
        self.wt = wt.strip();
        if (self.wt)==0: print("Error no wt _{}_ at {}".format(wt, pos))
        self.pos = pos;
        self.mut = mut.strip();
        if (self.mut)==0: print("Error no mut _{}_ at {}".format(mut, pos))
    def __str__(self):
        formatlist = []
        for _a in self.activity_list:
            formatlist.append("{:.1f}".format( _a ))
        return "{}{}{} {:.1f}/{} {}/{}".format(self.wt, self.pos, self.mut, self.avg_activity, self.std_activity, len(formatlist), formatlist)

class single_mut(single_mut_pred):
    def __init__(self, wt, pos, mut):
        super().__init__(wt, pos, mut)
        self.activity_list = []
        self.avg_activity = -1
        self.std_activity = -1;
    def add_actitivity(self, activity_str):
        self.activity_list.append(float(activity_str))
        self.avg_activity = np.mean(self.activity_list)
        self.std_activity = np.std(self.activity_list)
