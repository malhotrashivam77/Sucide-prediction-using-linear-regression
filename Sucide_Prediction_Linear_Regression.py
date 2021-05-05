#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("D:\\Datasets\\SucideRate\\SUCIDE PROJECT\\Suicides_in_India.csv")


# In[16]:


data.head(5)


# In[17]:


data.tail(5)


# In[4]:


D=data.groupby('Year').Total.sum()
D


# In[5]:


x=D.index.values.reshape(-1,1)
y=D.values


# In[6]:


plt.scatter(x,y)
plt.show()


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


reg=LinearRegression()


# In[9]:


reg.fit(x,y)


# In[10]:


reg.coef_


# In[11]:


reg.intercept_


# In[12]:


yp=reg.predict(x)


# In[13]:


plt.scatter(x,y)
plt.plot(x,yp)
plt.show()


# In[14]:


reg.score(x,y)*100


# In[15]:


reg.predict([[2021]])

