#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


cars=pd.read_csv(r"C:\Users\Anjishnu Satpathy\Documents\CarPrices\CarPrice_Assignment.csv")
cars.head()


# In[7]:


cars.info()


# In[9]:


A = cars[['citympg','price']]
A.tail()


# In[10]:


#converting pandas dataframe to numpy arrays
matrix= np.array(A.values, 'float')
matrix[0:5,:] #1st 5 rows 


# In[11]:


X = matrix[:,0]
y = matrix[:,1]
#input and target variable


# In[12]:


X=X/np.max(X)


# In[13]:


import matplotlib.pyplot as plt
plt.plot(X,y,'bo')
plt.ylabel('price')
plt.xlabel('citympg')
plt.legend(['PRICE'])
plt.title('citympg vs price')
plt.grid()
plt.show()


# In[14]:


def costfunc(x,y,theta):
    
    a = 1/(2*m)
    b = np.sum(((x@theta)-y)**2)
    j = (a)*(b)
    return j


# In[23]:


#initialising parameter
m = np.size(y)
X = X.reshape([205,1])
x = np.hstack([np.ones_like(X),X])
theta = np.zeros([2,1])
print(theta,'\n',m)


# In[24]:


print(costfunc(x,y,theta))


# In[31]:


def gradient(x,y,theta):
    alpha = 0.00001
    iteration = 2000
#gradient descend algorithm
    J_history = np.zeros([iteration, 1]);
    for iter in range(0,2000):
        error = (x @ theta) -y
        temp0 = theta[0] - ((alpha/m) * np.sum(error*x[:,0]))
        temp1 = theta[1] - ((alpha/m) * np.sum(error*x[:,1]))
        theta = np.array([temp0,temp1]).reshape(2,1)
        J_history[iter] = (1 / (2*m) ) * (np.sum(((x @ theta)-y)**2))   #compute J value for each iteration 
        return theta, J_history


# In[32]:


theta , J = gradient(x,y,theta)
print(theta)


# In[33]:


theta , J = gradient(x,y,theta)
print(J)


# In[38]:


#plot linear fit for our theta
plt.plot(X,y,'bo')
plt.plot(X,x@theta,'-')
plt.plot(X,y,'bo')
plt.ylabel('price')
plt.xlabel('citympg')
plt.legend(['PRICE_LINEARFIT'])
plt.title('citympg vs price')
plt.grid()
plt.show()


# In[39]:


predict1 = [1,(164/np.max(matrix[:,0]))] @ theta 
print(predict1)


# In[40]:





# In[ ]:




