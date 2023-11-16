#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten


# In[17]:


# Data import  form local computer  
X_train = np.loadtxt('Data\input.csv', delimiter = ',')
Y_train = np.loadtxt('Data\labels.csv', delimiter = ',')   
X_test = np.loadtxt('Data\input_test.csv', delimiter = ',')
Y_test = np.loadtxt('Data\labels_test.csv', delimiter = ',')


# In[18]:


print("Shape of X_train: ", X_train.shape)#2000 represents total no of images for training and 30000 features
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)


# In[19]:


X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)

X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)


# In[20]:


print("Shape of X_train: ", X_train.shape)#2000 represents total no of images for training and 30000 features
print("Shape of Y_train: ", Y_train.shape)
print("Shape of X_test: ", X_test.shape)
print("Shape of Y_test: ", Y_test.shape)


# In[21]:


X_train[1,:]


# In[22]:


#Rescale value between 0 and 1
X_train = X_train/255.0
X_test = X_test/255.0


# In[23]:


X_train[1,:]


# In[24]:


idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx])
plt.show()


# In[25]:


#BUIDLING MODEL
model = Sequential()
#adding layers
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)))#32 filters and (3,3)size of filter
model.add(MaxPooling2D((2,2)))#filter size

model.add(Conv2D(32, (3,3), activation = 'relu'))#
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))#fully connected layer 64 neuorns
model.add(Dense(1, activation = 'sigmoid'))#output layer  sigmoid beacuse binary classification


# In[26]:


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])#compile model
#as we have used binary classsification we will use binarycrossentropy loss function


# In[27]:


model.fit(X_train, Y_train, epochs = 5, batch_size = 64)


# In[28]:


model.evaluate(X_test, Y_test)


# In[29]:


#MAking Predicition
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if(y_pred == 0):
    pred = 'dog'
else:
    pred = 'cat'
    
print("Our model says it is a :", pred)

