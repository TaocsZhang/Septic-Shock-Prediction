# -*- coding: utf-8 -*-
# Python version: 3.6

import math
import time
import copy
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import random
from random import randint

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable


class CombineDataset(Dataset):
    def __init__(self, train_x, train_y):
        inputs = torch.from_numpy(train_x).type(torch.FloatTensor)
        labels = torch.from_numpy(train_y).long()
        self.x = inputs
        self.y = labels
       
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        item = self.x[idx]
        label = self.y[idx][47]
        return item, label
        
class SelectDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        x, label = self.dataset[self.idxs[item]]
        return x, label
    
class ClientUpdate(object):
    def __init__(self, client_dataset, config, idxs):
        self.train_loader = torch.utils.data.DataLoader(dataset=SelectDataset(config[client_dataset]['train_set'], idxs),
                                           batch_size=config[client_dataset]['batch_size'],
                                           shuffle=True, num_workers=0)
        self.learning_rate = config[client_dataset]['learning_rate']
        self.epochs = config[client_dataset]['epochs']

    def train(self, model):
        criterion = nn.CrossEntropyLoss()
        #optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, eps=1e-08, weight_decay=0)

        e_loss = []
        for epoch in range(1, self.epochs+1):

            train_loss = 0.0

            model.train()
            for data, labels in self.train_loader:

                if torch.cuda.is_available():
                    data, labels = data.cuda(), labels.cuda()

                # clear the gradients
                optimizer.zero_grad()
                # make a forward pass
                output = model(data)
                # calculate the loss
                loss = criterion(output, labels.view(len(labels)))
                #print(data.size(), output.size(), labels.size())
                # output.shape = 124*2, labels.shape = 124*1, torch.max(labels, 1)[1].shape = 124
                #loss = criterion(output, torch.max(labels, 1)[1]) 
                # do a backwards pass
                loss.backward()
                # perform a single optimization step
                optimizer.step()
                # update training loss
                train_loss += loss.item()*data.size(0)

            # average losses
            train_loss = train_loss/len(self.train_loader.dataset)
            e_loss.append(train_loss)

        total_loss = sum(e_loss)/len(e_loss)
        
        return model.state_dict(), total_loss