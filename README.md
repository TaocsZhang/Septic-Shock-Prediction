# Septic-Shock-Prediction
Use  LSTM model to predict septic shock in the Federated learning setting

Author: Tao Zhang

This is part of code for my intern project 'septic shock prediction'.

# Introduction
Septic shock is a life-threatening condition caused by a severe localised or system-wide infection. Sepsis is a major health concern with global estimates of 31.5 million cases per year. Case fatality rates are still unacceptably high, and early detection and treatment is vital since it significantly reduces mortality rates for this condition. Using machine learning models to make an early detecton is a promising solution.

One of the biggest challenges in medical tasks is that machine learning needs a large amount of data, while health data is highly sensitive data.
Federated learning is a methodology that can be used to mitigate many of the risks associated with sharing sensitive data. Federated learning is when the training of an algorithm is performed across multiple decentralized edge devices or servers containing local data samples, without any data exchange occurring. This means that external parties such as researchers and developers never need to access or see data in order to train and improve an algorithm. 

Due to data regulations, I cannot expose the model trained on multiple medical datasets, and I use a public dataset to train the model in a federated learning way. The dataset is partitioned into several parts, and each client updates the model on a subset of data. 

# Requirements
Python 3.6 <br>
Pytorch 1.2 <br>
Pandas <br>
Numpy <br>
Sklearn <br>

# Getting started
client_update.py is used to update the model in the local client. <br>

server_update use FedAvg algorithm to aggregate the final model. <br>

model.py is used to build LSTM model. <br>

utilis.py is used for data partition in IID and NON IID way. <br>

process_data.py and load_data.ipynb is used for data pre-processing. <br>

FL_main.py is used to train the federated model in different settings.

# Datasets
MIMIC datasets <br>

https://mimic.mit.edu/docs/gettingstarted/

# Evaluation
The results are given in FL_main.py. <br>

The results show that the LSTM model is able to give an early predition for septic shock with around 96% accuracy. <br>

The results show that the model performance in federated learning is pretty close to the model performance in centralized learning.
