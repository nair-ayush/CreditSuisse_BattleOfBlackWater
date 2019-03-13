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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn import metrics


# In[2]:


train = pd.read_csv('../Data/train.csv')


# In[18]:


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


# In[6]:


Y=train['bestSoldierPerc']
X = train.drop(['shipId','attackId','bestSoldierPerc','soldierId','friendlyKills','killRank'], axis=1)


# In[7]:


from sklearn import linear_model
lm=linear_model.LinearRegression()
model=lm.fit(X,Y)


# In[8]:


scores = cross_val_score(model, X, Y, cv=10)


# In[9]:


scores


# In[25]:


Xtest = test.drop(['shipId','attackId','soldierId','friendlyKills','killRank','index','bullshit'], axis=1)
ids=test['soldierId']


# In[ ]:





# In[27]:


Xtest.info()


# In[28]:


predictions = lm.predict(Xtest)


# In[36]:


pred=pd.DataFrame(predictions, columns=['bestSoldierPrec'])


# In[37]:


submission = pd.concat([ids,pred],axis=1)


# In[38]:


submission


# In[ ]:





# In[42]:


submission.to_csv('../sub1.csv')

