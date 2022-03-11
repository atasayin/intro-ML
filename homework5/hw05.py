#!/usr/bin/env python
# coding: utf-8

# # Engr421
# ## Homework 5
# ## Ata Sayın, 64437

# In[10]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def safelog2(x):
    if x == 0:
        return (0)
    else:
        return (np.log2(x))


# ## Data

# In[34]:


data_set=np.genfromtxt("hw04_data_set.csv",delimiter=",",skip_header=1)
X=data_set[:,0]
Y=data_set[:,1]


# In[35]:


X_train=X[:100]
xtest=X[100:133]
N_train=100
N_test=33


# In[36]:


y_train=Y[:100]
y_test=Y[100:133]


# ## Regression Tree

# In[59]:


P=15

# create necessary data structures
node_indices = {}
is_terminal = {}
need_split = {}
best_scores={}
node_splits = {}


# put all training instances into the root node
node_indices[1] = np.array(range(N_train))
is_terminal[1] = False
need_split[1] = True
split_nodes = [key for key, value in need_split.items() if value == True]


# In[60]:


while True:
    split_nodes = [key for key, value in need_split.items() if value == True]
    if len(split_nodes) == 0:
        break
    for split_node in split_nodes:
        data_indices = node_indices[split_node]
        need_split[split_node] = False
        
        if len((y_train[data_indices])) <= P:
            is_terminal[split_node] = True
        else:
            is_terminal[split_node] = False

            best_score = np.array([])
            best_splits = np.array([])
            
            unique_values = np.sort(np.unique(X_train[data_indices]))
            split_positions = (unique_values[1:len(unique_values)] + unique_values[0:(len(unique_values) - 1)]) / 2
            split_scores = np.repeat(0.0, len(split_positions))
            
            for s in range(len(split_positions)):
                left_indices = data_indices[X_train[data_indices] < split_positions[s]]
                right_indices = data_indices[X_train[data_indices] >= split_positions[s]]
                gm_left=np.mean(y_train[left_indices])
                gm_right=np.mean(y_train[right_indices])
                split_scores[s] =(np.sum((y_train[left_indices]-gm_left)**2)+
                np.sum((y_train[right_indices]-gm_right)**2))/len(data_indices)       
            
            best_score = np.min(split_scores)

            best_split = split_positions[np.argmin(split_scores)]
            node_splits[split_node] = best_split
            

            left_indices = data_indices[X_train[data_indices] < best_split]
            node_indices[2 * split_node] = left_indices
            is_terminal[2 * split_node] = False
            need_split[2 * split_node] = True
      
    
            right_indices = data_indices[X_train[data_indices] >= best_split]
            node_indices[2 * split_node + 1] = right_indices
            is_terminal[2 * split_node + 1] = False
            need_split[2 * split_node + 1] = True


# In[99]:


y_plot={}


# In[113]:


for node in node_indices:
    if node==None :
        pass
    if is_terminal:
        y_plot[node]=np.mean(y_train[node_indices[node]])


# In[123]:


y_plot


# In[128]:


plt.figure(figsize = (10, 6))
plt.plot(xtrain,ytrain,".b",markersize=10,label="training")
plt.plot(xtest,ytest,".r",markersize=10,label="test")
for n in range(len(node_splits)):
    plt.plot([node_splits[n], node_splits[n+1]], [y_plot[n], y_plot[n]], "k-")
for b in range(len(node_splits)-1):
    plt.plot([node_splits[n], node_splits[n]], [y_plot[b], y_plot[b + 1]], "k-")  
plt.legend(loc=2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# In[ ]:





# In[ ]:




