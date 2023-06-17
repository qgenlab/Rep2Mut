import os,sys
import pandas as pd
import math
import copy
import random

import numpy as np

import esm
import torch

import argparse

import pickle

WARNING_L = 3;
INFO_L = 2;
GEN_L = 1;



g_flanking_size = 400

output_level = WARNING_L

def read_fa(fafile):
    seq_dict = {}
    prev_id = ''
    prev_list = []
    hline_dict = {}
    with open(fafile, 'r') as mr:
        while True:
            line = mr.readline();
            if not line: break;
            line = line.strip();
            if len(line)==0: continue;

            if line[0] in ['>']:
                hline_dict[ line[1:].split()[0] ] = line
                if len(prev_list)>0:
                    seq_dict[ prev_id ] = ''.join( prev_list );
                    prev_id  = ''
                    prev_list = []
                prev_id = line[1:].split()[0];
            else:
                prev_list.append( line );
        if len(prev_list)>0:
            seq_dict[ prev_id ] = ''.join( prev_list );
    print("Read {}/{} seq.".format( len(seq_dict), len(hline_dict)  ))
    return (seq_dict, hline_dict)


def check_match(feat_ful_dict, seq_dict, input_excel_file, sheet_under_analysis):
    shnkeys = sheet_under_analysis
    shn_match_keys = []
    for shn in shnkeys:
        m_excel_data = pd.read_excel(input_excel_file, sheet_name=shn )
        matchbasen = [0, 0, 0];
        nomatch_pos = {}
        unmatchlist = []
        for mutinfo in m_excel_data[m_excel_data.columns[1]]:
            print(mutinfo)
            wt_base = mutinfo[0]
            mt_base = mutinfo[-1]
            if wt_base in ['X', '_'] or mutinfo in ['WT']: continue;

            try:
                mt_pos = int(mutinfo[1:-1])-1
            except:
                print("Warning!!! contain multiple mutations: {}/{} {} {}/{}".format( shn, feat_ful_dict[shn], mutinfo, matchbasen[0], matchbasen[1] ) )
                break;
            matchbasen[0] += 1
            if feat_ful_dict[shn][0] in seq_dict and mt_pos<len(seq_dict[ feat_ful_dict[shn][0] ]) and seq_dict[ feat_ful_dict[shn][0] ][ mt_pos ] == wt_base:
                matchbasen[1] += 1
            else:
                if mt_pos not in nomatch_pos:
                    unmatchlist.append( "{} {}/{}".format( mutinfo, mt_pos, seq_dict[ feat_ful_dict[shn][0] ][ mt_pos ] if feat_ful_dict[shn][0] in seq_dict and mt_pos<len(seq_dict[ feat_ful_dict[shn][0] ]) else "N" ))
                nomatch_pos[ mt_pos ] = False;
        if (matchbasen[0]==matchbasen[1]-matchbasen[2]) and matchbasen[0]>0:
            shn_match_keys.append( shn );
        else:
            if matchbasen[0]>0:
                print("Warning! not equal: {}/{} for {}/{}: {}".format( matchbasen[0], matchbasen[1], shn, feat_ful_dict[shn] , unmatchlist[:5] ))
        if output_level<=INFO_L: print("{}/{} {} {}/{}/{}".format( shn, feat_ful_dict[shn], len(shn_match_keys), matchbasen[0], matchbasen[1], m_excel_data.shape  ))
        sys.stdout.flush();

    return shn_match_keys


def read_mut_info( shn, seq_id, seq_dict, input_excel_file, batch_converter_alphabet_90_1, target_col):
    mut_vector_list = []
    
    m_excel_data = pd.read_excel(input_excel_file, sheet_name=shn)
    mut_target_list = []
    if target_col in m_excel_data.columns:
        is_col_in = True;
    else: 
        is_col_in = False;
        print("Not in feat_dict: {} {} {}".format( shn, target_col, m_excel_data.columns ))
    for row_id in range(m_excel_data.shape[0]):
        if is_col_in:
            if math.isnan(m_excel_data[ target_col ].iloc[row_id] ): continue;
            mut_target_list.append( m_excel_data[ target_col ].iloc[row_id] )
        else:
            if math.isnan(m_excel_data.iloc[row_id][2]): continue;
            mut_target_list.append( m_excel_data.iloc[row_id][2] )

        mutinfo = m_excel_data.iloc[row_id][1] 
        wt_base = mutinfo[0]
        mt_base = mutinfo[-1]
        if wt_base in ['X', '_'] or mutinfo in ['WT'] or mt_base in ['X', '_']: continue;

        mt_pos = int(mutinfo[1:-1])-1

        if mt_pos < g_flanking_size: rep_pos = mt_pos
        else: rep_pos = g_flanking_size

        start_ind = (mt_pos-g_flanking_size if mt_pos>g_flanking_size else 0) ;
        end___ind = mt_pos+g_flanking_size;

        mut_seq = []
        for _seqi in range(len(seq_dict[seq_id])):
            if _seqi in [ mt_pos ] : mut_seq.append( mt_base )
            else: mut_seq.append( seq_dict[seq_id][_seqi] )
        mut_data = [(0, seq_dict[seq_id][start_ind:(end___ind+1)]),
                    (1, (''.join(mut_seq))[start_ind:(end___ind+1)] ) ]
        _, _, batch_tokens = batch_converter_alphabet_90_1(mut_data)
        mut_vector_list.append((batch_tokens[0], batch_tokens[1], rep_pos, (m_excel_data.iloc[row_id][2] if not is_col_in else m_excel_data[ target_col ].iloc[row_id] ), mutinfo ) );

    mut_target_list = sorted(mut_target_list)
    print("Target value range{}/{}={} {}".format(shn,seq_id,  mut_target_list[:3], mut_target_list[-3:] ))
    return mut_vector_list

def get_rep_prottherm(model_90_1, mut_vector_list, device):
    wt_vector_list = []
    mt_vector_list = []
    y_vector_list = []
    mutinfo = []
    with torch.no_grad():
        model_90_1.to(device)
        for this_m in mut_vector_list:
           results = model_90_1(torch.stack((this_m[0],this_m[1])).to(device), repr_layers=[33])
           wt_vector_list.append(results['representations'][33][0][this_m[2]+1].detach().cpu())
           mt_vector_list.append(results['representations'][33][1][this_m[2]+1].detach().cpu())
           y_vector_list.append(torch.as_tensor([this_m[3]]))
           mutinfo.append(this_m[4])
        print("get_rep_prottherm", mutinfo[-1][-1])

    return wt_vector_list, mt_vector_list, y_vector_list, mutinfo





if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Generate ESM vectors from protein sequences and mutation data", formatter_class=argparse.ArgumentDefaultsHelpFormatter);

    parser.add_argument("-s", type=str, 
                                     help="The file containing the protein sequences. ");
    parser.add_argument("-n", type=str, 
                                     help="The name of the protein sequence in the file. ", default = None);
    parser.add_argument("-f", type=str, 
                                     help="The mapping file. ");
    parser.add_argument("-m", type=str, 
                                     help="The file containing the mutations, the position of the mutations and the output values.");
    parser.add_argument("-gpu", type=str, default="0",
                                     help="The device to use for the esm model. ");

  
    args = parser.parse_args()
    
    input_excel_file = args.m
    input_fa_file = args.s
    input_seqname = None
    if args.n != None : input_seqname = [ args.n ]
    m_excel_sheets = pd.ExcelFile(input_excel_file)
    print(len(m_excel_sheets.sheet_names), sorted(m_excel_sheets.sheet_names))
    seq_dict, hline_dict = read_fa(input_fa_file)
    print("Seq{}_ids: {}\n".format(len(seq_dict), sorted(list( seq_dict.keys() )) ) )
    map_file = args.f
    feat_excel = pd.read_excel(map_file)
    feat_ful_dict = {}
    for ir in range(feat_excel.shape[0]):
        feat_ful_dict[ feat_excel['Dataset'].iloc[ir] ] = (feat_excel['Seq_id'].iloc[ir], str(feat_excel['pred_column'].iloc[ir]  ))
    test_notini = 0;
    totali = 0;
    sheet_under_analysis = m_excel_sheets.sheet_names
    if args.n != None : sheet_under_analysis = input_seqname
    for shn in sheet_under_analysis:
        totali += 1;
        m_excel_data = pd.read_excel(input_excel_file, sheet_name=shn)
        if shn in feat_ful_dict:
            try:
                a = m_excel_data[ str(feat_ful_dict[shn][1]) ].iloc[0]
            except:
                print(totali, shn)
                print("\tCOL Not in {}".format( shn ), feat_dict[shn], m_excel_data.columns )
        else:
            test_notini += 1
            print(totali, shn)
            print("\t{} SHN Not in {}".format(test_notini, shn ))
    shn_match_keys = check_match(feat_ful_dict, seq_dict, input_excel_file, sheet_under_analysis)
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    model_90_1, alphabet_90_1 = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model_90_1 = model_90_1.to(device)
    batch_converter_alphabet_90_1 = alphabet_90_1.get_batch_converter()
    vectors = {}
    for shn in shn_match_keys:
        print("Traing_samples: {}".format(shn))
        sys.stdout.flush()
        try:
            mut_vector_list = read_mut_info( shn, feat_ful_dict[shn][0], seq_dict, input_excel_file, batch_converter_alphabet_90_1, feat_ful_dict[shn][1])
            print("Get mutation info: {}/{}".format(shn, len(mut_vector_list) ))
            sys.stdout.flush()
            wt_vector_list, mt_vector_list, y_vector_list, mutinfo = get_rep_prottherm(model_90_1, mut_vector_list, device)
            vectors[shn] = (wt_vector_list, mt_vector_list, y_vector_list, mutinfo)
        except Exception as e:
            logger.error('Failed to generate vectors: '+ str(e))
    pickle.dump(vectors, open("./vectors.p", 'wb'), protocol=pickle.HIGHEST_PROTOCOL)