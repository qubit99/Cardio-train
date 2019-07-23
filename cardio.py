#!/usr/bin/env python
# coding: utf-8

# In[203]:


import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[204]:


data = pd.read_csv('cardio_train.csv', sep=';', header=0)

# In[205]:


data.head()


# In[206]:


data = data.dropna()


# In[207]:


print(data.shape)


# In[208]:


print(data.columns)


# In[209]:


len(data.columns)


# In[250]:


data.drop(columns=['id'])


# In[251]:


data.cardio.value_counts()
sns.countplot(x='cardio', data=data, palette='hls')
plt.show()


# In[252]:


X = data[[
    'age',
    'gender',
    'height',
    'weight',
    'ap_hi',
    'ap_lo',
    'cholesterol',
    'gluc',
    'smoke',
    'alco',
    'active']]
y = data['cardio']


# In[273]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# In[274]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[598]:


y_pred = (logreg.predict_proba(X_test)[:, 1] >= 0.53).astype(bool)
y_pred


# In[599]:


cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[600]:


print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))


# In[601]:

# enter your parameters here
d = {'age': [16060], 'gender': [1], 'height': [157], 'weight': [69.0], 'ap_hi': [120], 'ap_lo': [62],
     'cholesterol': [2], 'gluc': [1], 'smoke': [0], 'alco': [0], 'active': [1]}
df = pd.DataFrame(data=d)


# In[602]:


logreg.predict(df)


# In[ ]:


# %%


# %%
