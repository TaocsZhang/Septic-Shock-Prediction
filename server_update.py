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


from client_update import *


def global_update(model, rounds, M, client_dataset, client_dict, config, plt_title, plt_color):
    """
    Function implements the Federated Averaging Algorithm from the FedAvg paper.
    Specifically, this function is used for the server side training and weight update

    Params:
    - model:           PyTorch model to train
    - rounds:          Number of communication rounds for the client update
    - M                Number of entities
    - E:               Number of training passes each client makes over its local dataset per round
    Returns:
    - model:           Trained model on the server
    """

    # global model weights
    global_weights = model.state_dict()

    # training loss
    train_loss = []

    # measure time
    start = time.time()

    for curr_round in range(1, rounds+1):
        w, local_loss = [], []

        for m in range(0, M):
            local_update = ClientUpdate(client_dataset, config, client_dict[m])
            weights, loss = local_update.train(model=copy.deepcopy(model))
            w.append(copy.deepcopy(weights))
            local_loss.append(copy.deepcopy(loss))
        print(local_loss)
        # updating the global weights
        weights_avg = copy.deepcopy(w[0])
        for k in weights_avg.keys():
            for i in range(1, len(w)):
                weights_avg[k] += w[i][k]
                
            weights_avg[k] = torch.div(weights_avg[k], len(w))

        global_weights = weights_avg

        # move the updated weights to our model state dict
        model.load_state_dict(global_weights)

        # loss
        loss_avg = sum(local_loss) / len(local_loss)
        print('Round: {}... \tAverage Loss: {}'.format(curr_round, round(loss_avg, 3)))
        train_loss.append(loss_avg)

    end = time.time()
    fig, ax = plt.subplots()
    x_axis = np.arange(1, rounds+1)
    y_axis = np.array(train_loss)
    ax.plot(x_axis, y_axis, 'tab:' + plt_color)

    ax.set(xlabel='Number of Rounds', ylabel='Train Loss', title=plt_title)
    ax.grid()
    fig.savefig('/home/ec2-user/SageMaker/save/' + plt_title + '.jpg', format='jpg')
    print("Training Done!")
    print("Total time taken to Train: {}".format(end-start))

    return model

def testing(model, dataset, config, criterion, num_classes, classes):
    #test_loss, total, correct = 0.0, 0.0, 0.0
    full_pred = []
    test_loss = 0.0
    correct_class = list(0. for i in range(num_classes))
    total_class = list(0. for i in range(num_classes))
    
    #test_loader = DataLoader(dataset, batch_size=bs)
    val_loader = torch.utils.data.DataLoader(dataset=config[dataset]['val_set'],
                                           batch_size=config[dataset]['batch_size'],
                                           shuffle=False, num_workers=0) # shuffle 
    l = len(val_loader)
    model.eval()
    for data, labels in val_loader:

        if torch.cuda.is_available():
            data, labels = data.cuda(), labels.cuda()

        output = model(data)
        loss = criterion(output, labels.view(len(labels)))
        test_loss += loss.item()*data.size(0)
        _, pred_labels = torch.max(output, 1)
        pred_labels = pred_labels.view(-1)
        
        full_pred.append(pred_labels)
        #correct += torch.sum(torch.eq(pred_labels, labels)).item()
        #total += len(labels)
        
        correct_tensor = pred_labels.eq(labels.data.view_as(pred_labels))
        #print(pred_labels, correct_tensor)
        
        if not torch.cuda.is_available():
            correct = np.squeeze(correct_tensor.numpy())
        else:
            np.squeeze(correct_tensor.cpu().numpy())
            
        #test accuracy for each object class
        for i in range(len(labels)):
            label = labels.data[i]
            correct_class[label] += correct[i].item()
            total_class[label] += 1
    
    # avg test loss
    test_loss = test_loss/len(val_loader.dataset)
    print("Test Loss: {:.6f}\n".format(test_loss))

    # print test accuracy
    for i in range(3):
        if total_class[i]>0:
            print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (classes[i], 100 * correct_class[i] / total_class[i],
                                                             np.sum(correct_class[i]), np.sum(total_class[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))

    print('\nFinal Test  Accuracy: {:.3f} ({}/{})'.format(
        100. * np.sum(correct_class) / np.sum(total_class),
        np.sum(correct_class), np.sum(total_class)))

    #accuracy = correct/total
    return full_pred
    