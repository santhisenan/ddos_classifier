#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, StandardScaler

from keras.utils import np_utils

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


data = pd.read_csv('balanced_dataset.csv')


# In[14]:


print(data.head())


# In[15]:


print(data.columns)


# In[16]:


data.drop("Unnamed: 0", axis=1, inplace=True)
data.drop("Unnamed: 0.1", axis=1, inplace=True)


# In[17]:


data.columns


# In[18]:


data[data.dtypes[(data.dtypes=="float64")|(data.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])


# In[19]:


dummy_pkt_type = pd.get_dummies(data['PKT_TYPE'])
data = pd.concat([data, dummy_pkt_type], axis = 1)

dummy_flags = pd.get_dummies(data['FLAGS'])
data = pd.concat([data, dummy_flags], axis = 1)

data.drop("NODE_NAME_FROM",axis=1,inplace=True)
data.drop("NODE_NAME_TO",axis=1,inplace=True)


# In[20]:


data.columns


# In[21]:


features = ['SRC_ADD', 'DES_ADD', 'PKT_ID', 'FROM_NODE', 'TO_NODE', 'PKT_SIZE',
       'FID', 'SEQ_NUMBER', 'NUMBER_OF_PKT', 'NUMBER_OF_BYTE', 'PKT_IN',
       'PKT_OUT', 'PKT_R', 'PKT_DELAY_NODE', 'PKT_RATE', 'BYTE_RATE',
       'PKT_AVG_SIZE', 'UTILIZATION', 'PKT_DELAY', 'PKT_SEND_TIME',
       'PKT_RESEVED_TIME', 'FIRST_PKT_SENT', 'LAST_PKT_RESEVED',
       'ack', 'cbr', 'ping', 'tcp', '-------', '---A---']
X = data[features].values
Y = data['PKT_CLASS']

print(X.shape)
print(Y.shape)


# In[22]:


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)


# In[25]:


dummy_Y = np_utils.to_categorical(encoded_Y)


# In[28]:


lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, encoded_Y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
X_new.shape


# In[29]:


scalar = StandardScaler(copy=True, with_mean=True, with_std=True)
scalar.fit(X_new)
standardised_X = scalar.transform(X_new)


# # variables available for export
# 
# *All values are numpy arrays*
# 
# - <strong>X</strong> : Initial X values
# - <strong>X_new</strong> : X after feature selection 
# - <strong>standardised_X</strong> : X_new after standardisation
# - <strong>Y</strong> : Intial Y with string values
# - <strong>encoded_Y</strong> : Y after integer encoding
# - <strong>dummy_Y</strong> : encoded_Y after OneHotEncoding
