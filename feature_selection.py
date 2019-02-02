#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('balanced_dataset.csv')


# In[4]:


print(data.head())
print(data.columns)


# In[8]:


data.drop("Unnamed: 0", axis=1, inplace=True)
data.drop("Unnamed: 0.1", axis=1, inplace=True)


# In[10]:


print(data.columns)
print(data.info())


# In[15]:


# data.drop("Unnamed: 0",axis=1,inplace=True)
# data.drop("Unnamed: 0.1",axis=1,inplace=True)

dummy_pkt_type = pd.get_dummies(data['PKT_TYPE'])
data = pd.concat([data, dummy_pkt_type], axis = 1)

dummy_flags = pd.get_dummies(data['FLAGS'])
data = pd.concat([data, dummy_flags], axis = 1)

data.drop("NODE_NAME_FROM",axis=1,inplace=True)
data.drop("NODE_NAME_TO",axis=1,inplace=True)


# In[20]:


data.columns


# In[26]:


features = ['SRC_ADD', 'DES_ADD', 'PKT_ID', 'FROM_NODE', 'TO_NODE',
       'PKT_SIZE', 'FID', 'SEQ_NUMBER', 'NUMBER_OF_PKT',
       'NUMBER_OF_BYTE', 'PKT_IN', 'PKT_OUT', 'PKT_R', 'PKT_DELAY_NODE',
       'PKT_RATE', 'BYTE_RATE', 'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_DELAY',
       'PKT_SEND_TIME', 'PKT_RESEVED_TIME', 'FIRST_PKT_SENT',
       'LAST_PKT_RESEVED','ack', 'cbr', 'ping', 'tcp', '-------',
       '---A---']
X = data[features].values
Y = data['PKT_CLASS']

print(X.shape)
print(Y.shape)


# In[35]:


scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar.fit(X)
standardised_X = scalar.transform(X)


# In[27]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# In[36]:


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(standardised_X, Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(standardised_X)
X_new.shape


# In[37]:


print(X.shape)
print(X_new.shape)


# In[ ]:




