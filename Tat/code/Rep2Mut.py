import torch
import esm
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


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
            self.outputs[e] = nn.Linear(self.hidden_dim1 + self.pos_dim, self.hidden_dim2, dtype=torch.float64)
    def forward(self, wt_x, mt_x, pos_x, output = '0'):
        wt_x = self.act1(self.wt_linear1(wt_x))
        wt_x = self.wt_dropout_lary(wt_x)
        mt_x = self.act2(self.mt_linear1(mt_x))
        mt_x = self.mt_dropout_lary(mt_x)
        if output=="Tat":a_x = torch.sigmoid(self.outputs[output]( torch.cat((wt_x   *  mt_x, pos_x),1)));
        else: a_x = self.outputs[output]( torch.cat((wt_x   *  mt_x, pos_x),1));
        return a_x