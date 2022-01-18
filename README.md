# Septic Shock Prediction
Use  LSTM model to predict septic shock via Federated learning

Author: Tao Zhang

This is part of code for my intern project 'Septic Shock Prediction'.

# Introduction
Septic shock is a life-threatening condition caused by a severe localised or system-wide infection. Sepsis is a major health concern with global estimates of 31.5 million cases per year. Case fatality rates are still unacceptably high, and early detection and treatment is vital since it significantly reduces mortality rates for this condition. Using machine learning models to make an early detecton is a promising solution.

One of the biggest challenges in medical tasks is that machine learning needs a large amount of data, while health data is highly sensitive data. Federated learning is a methodology that can be used to mitigate many of the risks associated with sharing sensitive data. Federated learning is when the training of an algorithm is performed across multiple decentralized edge devices or servers containing local data samples, without any data exchange occurring. This means that external parties such as researchers and developers never need to access or see data in order to train and improve an algorithm. 

Due to data regulations, I cannot expose the model trained on multiple medical datasets, and I use a public dataset to train the model in a federated learning way. The dataset is partitioned into several parts, and each client updates the model on a subset of data. 

## Method
FedAvg is used as the federated learning algorithm in the project, which is a popular method of FL. In each round, first, the server sends the global model to randomly selected participants. Second, each party uses its local data set to update the model. Then, the updated model is sent back to the server. Finally, the server averages the received local model to the updated global model. Different from traditional distributed SGD, all parties use multiple eras to update their local models, which can reduce the number of communication rounds and improve communication efficiency.


# Requirements
Python 3.6 <br>
Pytorch 1.2 <br>
Pandas 1.2.5<br>
Numpy 1.2.0<br>
Sklearn 1.0.0<br>

# Getting started
client_update.py is used to update the model in the local client. <br>

server_update use FedAvg algorithm to aggregate the final model. <br>

model.py is used to build LSTM model. <br>

utilis.py is used for data partition in IID and NON IID way. <br>

process_data.py and load_data.ipynb are used for data pre-processing. <br>

FL_main.ipynb is used to train the federated model in different settings.

# Datasets
MIMIC datasets <br>

The dataset we used is MIMIC-III is a large, freely-available database comprising de-identified health-related data associated with over forty thousand patients. After data processing, there are 13 features, including age, temperature, heart rate, blood lactate, and so on.  We select the sequence of each patient’s data in the last 48 hours before the diagnosis. If the diagnosis is having septic shock, the last 48-hour data are all marked as True.  If the diagnosis is not having septic shock, the last 48-hour data are all marked as False.  if the sequence of the patient’s data is less than 48 hours, the padding -1 is used to fill in the sequence.

https://mimic.mit.edu/docs/gettingstarted/

# Evaluation
The results are given in FL_main.ipynb. <br>

The goal of the experiment is to verify federated learning is able to achieve similar model performance compared with the centralized training in the septic shock task. To assess this, I conduct the experiments in three different settings: centralised mode, federated learning with IID data distribution, and federated learning with Non-IID data distribution. In the centralised mode, the model is trained on the entire dataset. The result of the centralised model can be regarded as the upper performance that federated learning is likely to reach. 

In the IID data distribution mode, data distribution in different entities follows a similar data distribution. In the Non-IID data distribution mode, data distribution in different entities follows different data distributions. For example, different regions may have very different disease distributions. Due to the ozone hole, countries in the southern hemisphere may have more skin cancer patients than in the northern hemisphere. Then, the label distribution differs for each party.

Two types of model outputs are considered. Single output: the model only gives the final output of a sequence. For example, when the input is 48-hour information of all patients, the output is the last hour prediction of whether the patient has septic shock in the last hour. multiple outputs: the model gives each hour output of a sequence. For example, when the input is 48-hour information of all patients, the output is the 48-hour predictions of whether the patient has septic shock in the last hour. 

In the centralised mode, the accuracy is around 96%. in the IID data distribution mode, accuracy is around 96%. in the Non-IID data distribution mode, accuracy is around 88%.  I also calculate the accuracy for each class and the confusion matrix for each mode. As we can see, federated learning is able to achieve similar model performance compared with the centralised mode. Though there is a little drop in model performance, the model performance will be better when more entities participate in the training. I have also implemented a Transformer model in the septic shock task. However, the model performance is not as good as LSTM. I think transformer may not be the best choice for the task, and the other possibility is that model parameters need more time to optimise.
