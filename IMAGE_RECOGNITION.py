#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING LIBRARIES

# In[1]:


import os
import warnings
warnings.simplefilter('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from skimage.io import imread,imshow
from skimage.transform import resize
from skimage.color import rgb2gray


# ### SETTING WORKING DIRECTORY 

# In[3]:


Prithiviraj=os.listdir("C:/Users/janan/Pictures/Prithiviraj")
Sakshi=os.listdir("C:/Users/janan/Pictures/sakshi")
Selin=os.listdir("C:/Users/janan/Pictures/selin")


# ### READING THE IMAGES

# In[4]:


limit = 40
prithiviraj_images = [None]*limit
j=0

for i in Prithiviraj:
    if(j<limit):
        prithiviraj_images[j]=imread("C:/Users/janan/Pictures/Prithiviraj/"+i)
        j+=1
    else:
        break


# In[5]:


sakshi_images = [None]*limit
j=0

for i in Sakshi:
    if(j<limit):
        sakshi_images[j]=imread("C:/Users/janan/Pictures/sakshi/"+i)
        j+=1
    else:
        break


# In[6]:


selin_images = [None]*limit
j=0

for i in Selin:
    if(j<limit):
        selin_images[j]=imread("C:/Users/janan/Pictures/selin/"+i)
        j+=1
    else:
        break


# ### VIEWING THE IMAGE

# In[7]:


imshow(prithiviraj_images[15])


# In[8]:


imshow(sakshi_images[19])


# In[9]:


imshow(selin_images[35])


# In[10]:


imshow(sakshi_images[10])


# ### CONVERTING COLOR IMAGES TO GRAY SCALE

# In[11]:


prithiviraj_gray= [None]*limit
j=0

for i in Prithiviraj:
    if(j<limit):
        prithiviraj_gray[j]=rgb2gray(prithiviraj_images[j])
        j+=1
    else:
        break


# In[12]:


sakshi_gray= [None]*limit
j=0

for i in Sakshi:
    if(j<limit):
        sakshi_gray[j]=rgb2gray(sakshi_images[j])
        j+=1
    else:
        break


# In[13]:


selin_gray= [None]*limit
j=0

for i in Selin:
    if(j<limit):
        selin_gray[j]=rgb2gray(selin_images[j])
        j+=1
    else:
        break


# ### VIEWING THE GRAY SCALE IMAGES

# In[14]:


imshow(prithiviraj_gray[4])


# In[15]:


imshow(sakshi_gray[32])


# In[16]:


imshow(selin_gray[15])


# #### CHECKING THE SIZE OF THE IMAGE

# In[17]:


prithiviraj_gray[4].shape


# In[18]:


sakshi_gray[9].shape


# In[19]:


selin_gray[5].shape


# #### MATRIX RESIZING

# In[20]:


for j in range(40):
    pr = prithiviraj_gray[j]
    prithiviraj_gray[j]=resize(pr,(512,512))


# In[21]:


for j in range(40):
    sk = sakshi_gray[j]
    sakshi_gray[j]=resize(sk,(512,512))


# In[22]:


for j in range(40):
    sl = selin_gray[j]
    selin_gray[j]=resize(sl,(512,512))


# ### VIEWING THE RESIZED IMAGE

# In[23]:


imshow(prithiviraj_gray[4])


# In[24]:


imshow(sakshi_gray[18])


# In[25]:


imshow(selin_gray[8])


# ### IMAGE MATRIX TO VECTOR CONVERSION

# ##### FOR PRITHIVIRAJ

# In[26]:


len_of_prithiviraj= len(prithiviraj_gray)
len_of_prithiviraj


# In[27]:


image_size_prithiviraj = prithiviraj_gray[2].shape
image_size_prithiviraj


# In[28]:


flatten_size_prithiviraj = image_size_prithiviraj[0]*image_size_prithiviraj[1]
flatten_size_prithiviraj


# In[29]:


for i in range(len_of_prithiviraj):
    prithiviraj_gray[i]=np.ndarray.flatten(prithiviraj_gray[i]).reshape(flatten_size_prithiviraj,1)


# In[30]:


prithiviraj_gray = np.dstack(prithiviraj_gray)
prithiviraj_gray


# In[31]:


prithiviraj_gray = np.rollaxis(prithiviraj_gray,axis=2,start=0)
prithiviraj_gray.shape


# In[32]:


prithiviraj_gray = prithiviraj_gray.reshape(len_of_prithiviraj,flatten_size_prithiviraj)
prithiviraj_gray.shape


# In[33]:


prithiviraj_df = pd.DataFrame(prithiviraj_gray)
prithiviraj_df


# In[34]:


prithiviraj_df["label"] = "Prithiviraj"
prithiviraj_df


# #### FOR SAKSHI

# In[35]:


len_of_sakshi = len(sakshi_gray)
len_of_sakshi


# In[36]:


image_size_sakshi = sakshi_gray[2].shape
image_size_sakshi


# In[37]:


flatten_size_sakshi = image_size_sakshi[0]*image_size_sakshi[1]
flatten_size_sakshi


# In[38]:


for i in range(len_of_sakshi):
    sakshi_gray[i]=np.ndarray.flatten(sakshi_gray[i]).reshape(flatten_size_sakshi,1)


# In[39]:


sakshi_gray = np.dstack(sakshi_gray)
sakshi_gray


# In[40]:


sakshi_gray = np.rollaxis(sakshi_gray,axis=2,start=0)
sakshi_gray.shape


# In[41]:


sakshi_gray = sakshi_gray.reshape(len_of_sakshi,flatten_size_sakshi)
sakshi_gray.shape


# In[42]:


sakshi_df = pd.DataFrame(sakshi_gray)
sakshi_df


# In[43]:


sakshi_df["label"] = "Sakshi"
sakshi_df


# #### FOR SELIN

# In[44]:


len_of_selin = len(selin_gray)
len_of_selin


# In[45]:


image_size_selin= selin_gray[2].shape
image_size_selin


# In[46]:


flatten_size_selin = image_size_selin[0]*image_size_selin[1]
flatten_size_selin


# In[47]:


for i in range(len_of_selin):
    selin_gray[i]=np.ndarray.flatten(selin_gray[i]).reshape(flatten_size_selin,1)


# In[48]:


selin_gray = np.dstack(selin_gray)
selin_gray


# In[49]:


selin_gray = np.rollaxis(selin_gray,axis=2,start=0)
selin_gray.shape


# In[50]:


selin_gray = selin_gray.reshape(len_of_selin,flatten_size_selin)
selin_gray.shape


# In[51]:


selin_df = pd.DataFrame(selin_gray)
selin_df


# In[52]:


selin_df["label"] = "Selin"
selin_df


# ### COMBINING ALL THREE DATAFRAMES

# In[53]:


combine = pd.concat([prithiviraj_df,sakshi_df])
combine


# In[54]:


completedata = pd.concat([combine,selin_df])
completedata


# In[55]:



from sklearn.utils import shuffle


# In[56]:


people_indexed = shuffle(completedata).reset_index()
people_indexed


# In[57]:


people = people_indexed.drop(['index'],axis=1)
people


# #### INITIALIZE DEPENDENT VARIABLE AND INDEPENDENT VARIABLE

# In[58]:


x = people.values[:,:-1]
y = people.values[:,-1]


# In[59]:


x


# In[60]:


y


# #### SPLITTING THE DATASET

# In[61]:


from sklearn.model_selection import train_test_split


# In[62]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# ### SVM

# In[63]:


from sklearn import svm


# In[64]:


clf = svm.SVC()
clf.fit(x_train,y_train)


# ### IMAGE PREDICTION

# In[65]:


y_pred = clf.predict(x_test)
y_pred


# ### ACCURACY

# In[66]:


from sklearn import metrics


# In[67]:


accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy


# In[68]:


from sklearn.metrics import confusion_matrix


# In[69]:


confusion_matrix(y_test,y_pred)


# In[ ]:




