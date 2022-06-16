#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np


# In[28]:


pip install pyttsx3


# In[31]:


import cv2


# In[32]:


import pyttsx3 as p


# In[33]:


engine = p.init()


# In[34]:


with_mask = np.load('with_mask.npy')
without_mask  = np.load('without_mask.npy')


# In[35]:


with_mask.shape


# In[36]:


without_mask.shape


# In[37]:


with_mask = with_mask.reshape(221,50*50*3)


# In[38]:


without_mask = without_mask.reshape(221,50*50*3)


# In[39]:


with_mask.shape


# In[40]:


X = np.r_[with_mask,without_mask]


# In[41]:


X.shape


# In[42]:


labels = np.zeros(X.shape[0])


# In[43]:


labels


# In[44]:


labels[221:]= 1.0


# In[45]:


names = {0: 'Mask', 1: 'No Mask'}


# In[46]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


x_train, x_test, y_train, y_test =  train_test_split(X,labels, test_size=0.25)


# In[49]:


x_train.shape


# In[50]:


from sklearn.decomposition import PCA


# In[51]:


pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)


# In[52]:


x_train[0]


# In[53]:


x_train.shape


# In[54]:


svm = SVC()
model = svm.fit(x_train, y_train)  


# In[55]:


y_train.shape


# In[56]:


x_test = pca.transform(x_test)
y_pred = svm.predict(x_test)


# In[57]:


accuracy_score(y_test,y_pred)


# In[58]:


haar_data = cv2.CascadeClassifier('data.xml')
capture = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face, (50,50))
            face = face.reshape(1,-1)
            face = pca.transform(face)
            pred = svm.predict(face)
            if int(pred) == 1:
                engine.say("You haven't worn your mask please wear it")
                engine.runAndWait()
            n = names[int(pred)]
            cv2.putText(img, n, (x,y), font, 1,(224,250,250), 2)
            print(n)
        cv2.imshow('Result',img)
        if cv2.waitKey(2) == 27 or len(data) > 220:
            break
            
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




