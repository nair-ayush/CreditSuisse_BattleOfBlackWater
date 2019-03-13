#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics


# In[2]:


train = pd.read_csv('../Data/train.csv')


# In[3]:


test = pd.read_csv('../Data/test.csv')


# In[ ]:


train.head()


# In[ ]:


print(train.shape)
print(train.dtypes)
print(train.columns)


# In[ ]:


train.describe()


# In[ ]:


correlation = train.corr(method='pearson')
columns = correlation.nlargest(25, 'bestSoldierPerc').index
fig, ax = plt.subplots(figsize=(15,15))
correlation_map = np.corrcoef(train[columns].values.T)
sns.set(font_scale=1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, fmt='.2f', yticklabels=columns.values, xticklabels=columns.values)

plt.show()


# In[4]:


Y=train['bestSoldierPerc']
X = train.drop(['shipId','attackId','bestSoldierPerc','soldierId','friendlyKills','killRank'], axis=1)


# In[5]:


from sklearn import linear_model
scaler = StandardScaler().fit(X)
rescaled_X_train = scaler.transform(X)
lm=linear_model.LinearRegression()
model=lm.fit(X,Y)


# In[8]:


scores = cross_val_score(model, X, Y, cv=10)


# In[9]:


scores


# In[6]:


Xtest = test.drop(['shipId','attackId','soldierId','friendlyKills','killRank','index','bullshit'], axis=1)
ids=test['soldierId']


# In[7]:


rescaled_X_test = scaler.transform(Xtest)
predictions = model.predict(rescaled_X_test)


# In[8]:


Xtest.info()


# In[9]:


predictions = lm.predict(Xtest)


# In[10]:


pred=pd.DataFrame(predictions, columns=['bestSoldierPrec'])


# In[11]:


submission = pd.concat([ids,pred],axis=1)


# In[12]:


submission


# In[ ]:





# In[13]:


submission.to_csv('../sub2.csv',index=False)

