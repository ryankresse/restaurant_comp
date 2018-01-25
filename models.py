import torch
import re, pickle, collections, numpy as np, math, operator, pdb
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
from torch.autograd import Variable
import pdb

class EmbedModel(nn.Module):
    def __init__(self, embed_info, do=0.9):
        super(EmbedModel, self).__init__()
        for i, info in enumerate(embed_info):
            #pdb.set_trace()
            self.add_module('embed_{}'.format(i), nn.Embedding(int(info[1]), int(info[2])))
        self.embs = list(self.children())
        self.do = nn.Dropout(p=do)
    def forward(self, x):
        out = []
        for i in range(x.size()[1]):
            #print(self.embs[i])
            out.append(self.embs[i](x[:, i]))
        return self.do(torch.cat(out, 1))
        

class ContModel(nn.Module):
    def __init__(self, inp_size, out_size, do=0.9):
        super(ContModel, self).__init__()
        self.linear = nn.Linear(inp_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.do = nn.Dropout(p=do)
        
    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.bn(x)
        return self.do(x) #SHOULD WE HAVE A NON-LINEARITY HERE?

class RegModel(nn.Module):
    def __init__(self, inp_size, h1_size=512, h2_size=1024, do=0.9):
        super(RegModel, self).__init__()
        self.h1 = nn.Linear(inp_size, h1_size)
        self.bn1 = nn.BatchNorm1d(h1_size)
        self.do1 = nn.Dropout(p=do)
        
        self.h2 = nn.Linear(h1_size, h2_size)
        self.bn2 = nn.BatchNorm1d(h2_size)
        self.do2 = nn.Dropout(p=do)
        self.out = nn.Linear(h2_size, 1)
            
    def forward(self, x):
        x = F.relu(self.h1(x))
        x = self.bn1(x)
        x = self.do1(x)
        x = F.relu(self.h2(x))
        x = self.bn2(x)
        x = self.do2(x)
        return self.out(x)
