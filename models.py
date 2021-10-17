# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        #self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.3)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
       
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
    
        # Propagate input through LSTM
        # h_out is the last hidden state of the sequences
        # h_out = (1 x batch_size x hidden_dim)
        # ht[-1] = (batch_size x hidden_dim)
        output, (h_out, c_out) = self.lstm(x, (h_0, c_0))
        
        #output = self.fc(output)
        #output = F.softmax(output)
        #h_out = h_out.view(-1, self.hidden_size) 
        out = self.fc(h_out[-1]) 
        #print(out.shape) # 124*3
        probs = F.softmax(out, dim=1)
        #print(probs)
        return probs
