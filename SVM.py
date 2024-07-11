#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[4]:


df = pd.read_csv('../ML/data/breast-cancer-wisconsin.data.csv')
df['1.3'] = pd.to_numeric(X['1.3'], errors='coerce')
df = df.dropna()
X = df.drop(['1000025', '2.1'], axis=1)
y = df['2.1']


y.shape


# In[126]:


clf = svm.SVC(kernel='rbf') # Linear, Polynomial, or RBF Kernel


# In[124]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
y_train


# In[125]:


clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

