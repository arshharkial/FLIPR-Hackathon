#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
warnings.filterwarnings("ignore")
sns.set()
np.set_printoptions(threshold = sys.maxsize)


# ## Import Dataset

# In[2]:


dataset = pd.read_excel("Train_dataset.xlsx")
dataset.head()


# ## Check nan

# In[3]:


print(dataset.isna().any())


# ## Droping Stock Index from string to int

# In[4]:


dataset = dataset.drop(['Stock Index'], axis = 1)


# ## Creating independent and dependent variable

# In[5]:


x = dataset.iloc[ : , : -1].values
y = dataset.iloc[ : , -1].values


# ## Creating train-test split

# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.1, random_state = 0)


# ## Simple Imputer

# In[7]:


from sklearn.impute import SimpleImputer
imputer_train = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer_train.fit(x_train[ : , 2 :])
x_train[ : , 2 :] = imputer_train.transform(x_train[ : , 2 :])

imputer_valid = SimpleImputer(missing_values = np.nan, strategy='mean')
imputer_valid.fit(x_valid[ : , 2 :])
x_valid[ : , 2 :] = imputer_valid.transform(x_valid[ : , 2 :])


# ## Feature Scaling

# In[8]:


from sklearn.preprocessing import StandardScaler
standard_scaler_train = StandardScaler()
x_train[ : , 2 : ] = np.array(standard_scaler_train.fit_transform(x_train[ : , 2 : ]))

standard_scaler_valid = StandardScaler()
x_valid[ : , 2 : ] = np.array(standard_scaler_valid.fit_transform(x_valid[ : , 2 : ]))


# ## One Hot Enocding Categorical Variables

# In[9]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
column_transformer_train = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1])],
                                             remainder = 'passthrough')
x_train = np.array(column_transformer_train.fit_transform(x_train))

column_transformer_valid = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1])],
                                             remainder = 'passthrough')
x_valid = np.array(column_transformer_valid.fit_transform(x_valid))


# ## Creating Gradient Boosting Regression Model

# In[53]:


from sklearn.ensemble import GradientBoostingRegressor
regressor_gb = GradientBoostingRegressor(loss = 'huber',
                                         n_estimators = 100,
                                         random_state = 42)
regressor_gb.fit(x_train, y_train)
y_pred_gb = regressor_gb.predict(x_train)


# In[54]:


residuals = y_train - y_pred_gb
sns.distplot(residuals)


# In[55]:


rms_gb = np.sqrt(np.mean(np.power((np.array(y_train)-np.array(y_pred_gb)),2)))
print(rms_gb)
ms_gb = np.sum(np.power((np.array(y_train)-np.array(y_pred_gb)), 2))/len(y_train)
print(ms_gb)


# In[56]:


y_pred_valid_gb = regressor_gb.predict(x_valid)
residuals_valid = y_valid - y_pred_valid_gb
sns.distplot(residuals_valid)
rms_valid_gb = np.sqrt(np.mean(np.power((np.array(y_valid)-np.array(y_pred_valid_gb)),2)))
print(rms_valid_gb)
ms_valid_gb = np.sum(np.power((np.array(y_valid)-np.array(y_pred_valid_gb)), 2))/len(y_valid)
print(ms_valid_gb)


# ## Test set data preprocessing

# In[57]:


test_set = pd.read_excel("Test_dataset.xlsx")


# In[58]:


stock_index = test_set.iloc[ : , 0].values
test_set = test_set.drop(['Stock Index'], axis = 1)
print(test_set.isna().any())
test_set = test_set.fillna(dataset.mean())


# In[59]:


print(test_set.isna().any())


# In[60]:


x_test = test_set.iloc[ : , : ].values


# In[61]:


standard_scaler_test = StandardScaler()
x_test[ : , 2 : ] = np.array(standard_scaler_test.fit_transform(x_test[ : , 2 : ]))


# In[62]:


column_transformer_test = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0, 1])],
                                             remainder = 'passthrough')
x_test = np.array(column_transformer_test.fit_transform(x_test))


# ## Test set prediction

# In[63]:


y_test_pred_gb = regressor_gb.predict(x_test)


# In[64]:


print(y_test_pred_gb)


# ## Creating Output File

# In[65]:


import csv
data = {'Stock Index' : stock_index, 'Stock Price' : y_test_pred_gb}
output = pd.DataFrame(data = data)
output.to_csv("01.csv", index = False)
    

