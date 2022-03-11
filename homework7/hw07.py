#!/usr/bin/env python
# coding: utf-8

# # Engr421 
# ## Homework 7
# ##  Linear Discriminant Analysis
# ### Ata Sayın, 64437

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd
import scipy.spatial as spa


# In[2]:


# read data into memory
X_train = np.genfromtxt("hw07_training_images.csv", delimiter = ",")
y_train = np.genfromtxt("hw07_training_labels.csv", delimiter = ",").astype(int)
X_test = np.genfromtxt("hw07_test_images.csv", delimiter = ",")
y_test = np.genfromtxt("hw07_test_labels.csv", delimiter = ",").astype(int)

N=X_train.shape[0]
# get number of classes and number of features
C = np.max(y_train)
D = X_train.shape[1]


# In[3]:


#sample means
sample_means=np.array([np.mean(X_train[y_train==c+1],axis=0) for c in range(C)])
sample_means.shape


# In[4]:


#sample covariance
sample_covariance=np.array([np.cov(np.transpose(X_train[y_train==c+1])) for c in range(C)])
sample_covariance.shape


# In[5]:


#Within class matrix
Sw=np.sum(sample_covariance,axis=0)+np.diag(np.repeat(1e-10,D))
Sw.shape


# In[6]:


#total sample of the whole data
total_sample=np.array(np.mean(X_train,axis=0))
total_sample.shape


# In[7]:


#between class scatter matrix 
Y=np.zeros((y_train.shape[0], C)).astype(int)
Y[range(y_train.shape[0]), y_train-1] = 1


# In[8]:


Sb=np.sum([np.dot((sample_means[c]-total_sample).reshape(784,1),((sample_means[c]-total_sample).reshape(784,1)).T
                )*np.sum(Y,axis=0)[c] for c in range(C)]
         ,axis=0)
Sb.shape


# In[9]:


J=np.linalg.inv(Sw).dot(Sb)


# In[10]:


w,v=linalg.eigh(J)
v=np.real(v)
w=np.real(w)


# In[11]:


#First largest
v1=v[:, np.argsort(-w)[0]]
v1


# In[12]:


#Second largest
v2=v[:, np.argsort(-w)[1]]
v2


# In[13]:


W=np.transpose(np.vstack((v1,v2)))
W.shape


# In[14]:


Z_train=np.matmul(W.T,X_train.T)
Z_train.shape


# ## Plot

# In[15]:


plt.figure(figsize = (10, 6))
plt.plot(Z_train[0][y_train==1],Z_train[1][y_train==1],"r.", markersize = 5,label="T-Shirt")
plt.plot(Z_train[0][y_train==2],Z_train[1][y_train==2],"b.", markersize = 5,label="Trouser")
plt.plot(Z_train[0][y_train==3],Z_train[1][y_train==3],"g.", markersize = 5,label="Dress")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.show()


# In[16]:


c1=X_train[y_train==+1].reshape(1012,28,28)[0]
c2=X_train[y_train==+2].reshape(974,28,28)[0]
c3=X_train[y_train==+3].reshape(1014,28,28)[0]


# In[17]:


fig, (ax1,ax2,ax3) = plt.subplots(1, 3)
ax1.imshow(c1,cmap='binary_r')
ax1.axis('off')
ax2.imshow(c2,cmap='binary_r')
ax2.axis('off')
ax3.imshow(c3,cmap='binary_r')
ax3.axis('off')
plt.show()


# ## Five-nearest Neighboor Classifier

# In[18]:


K=5


# In[19]:


#Train data
Dis = spa.distance_matrix(Z_train.T,Z_train.T)
Dis[np.argsort(Dis)[:,K]]

k_distances=[]
k_distances=np.vstack([y_train[np.argsort(Dis)[:,k+1]] for k in range(K)])    

p_hat=np.array([np.sum(np.array(k_distances==c+1),axis=0) for c in range(C)])/K
print(p_hat)


# In[20]:


y_predicted = np.argmax(p_hat, axis = 0) + 1
confusion_matrix = pd.crosstab(y_predicted, y_train, rownames = ['y_pred'], colnames = ['y_train'])
print(confusion_matrix)


# In[21]:


#Test data
Z_test=np.matmul(W.T,X_test.T)

Dis = spa.distance_matrix(Z_test.T,Z_test.T)
Dis[np.argsort(Dis)[:,K]]

k_distances=[]
k_distances=np.vstack([y_test[np.argsort(Dis)[:,k+1]] for k in range(K)])

p_hat=np.array([np.sum(np.array(k_distances==c+1),axis=0) for c in range(C)])/K
print(p_hat)


# In[22]:


y_predicted = np.argmax(p_hat, axis = 0) + 1
confusion_matrix = pd.crosstab(y_predicted, y_test, rownames = ['y_pred'], colnames = ['y_test'])
print(confusion_matrix)

