#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("PCOS_data_without_infertility1.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.drop('Sl. No', axis = 1,inplace = True)
df.drop('Patient File No.', axis = 1,inplace = True)
df.drop('BMI', axis = 1,inplace = True)
df.drop('Unnamed: 44', axis = 1,inplace = True)
df.drop('Waist:Hip Ratio', axis = 1,inplace = True)


# In[5]:


df['AMH(ng/mL)'] = pd.to_numeric(df['AMH(ng/mL)'],errors='coerce')
df['II    beta-HCG(mIU/mL)'] = pd.to_numeric(df['II    beta-HCG(mIU/mL)'],errors='coerce')
df['FSH/LH'] = pd.to_numeric(df['FSH/LH'],errors='coerce')


# In[6]:


df['FSH/LH'] = df['FSH(mIU/mL)']/df['LH(mIU/mL)']


# In[7]:


df = df.dropna()


# In[8]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[9]:


x = df[[ ' Age (yrs)','Height(Cm) ','Weight (Kg)','Cycle(R/I)','Cycle length(days)','Pregnant(Y/N)','  I   beta-HCG(mIU/mL)','II    beta-HCG(mIU/mL)','FSH/LH','AMH(ng/mL)','Weight gain(Y/N)']]
y = df['PCOS (Y/N)']


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state =14)


# In[16]:


model = LogisticRegression(class_weight = 'balanced', max_iter =1500)


# In[17]:


model.fit(x_train, y_train)


# In[18]:


model.score(x_test,y_test)


# In[19]:


model.predict([[23,165,69,4,7,0,1.99,1.99,3,2,1]])


# In[20]:


model.predict([[23,165,60,4,7,0,1.99,1.99,3,2,0]])


# In[22]:


model.predict([[20,155,80,2,9,0,1.99,1.99,3,21,0]])


# In[23]:


import pickle
pickle.dump(model, open('pcos2.pkl', 'wb'))

