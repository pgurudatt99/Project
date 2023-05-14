#!/usr/bin/env python
# coding: utf-8

# In[3]:


#1.Learn to use Cross validation to pick the best models.
import pandas as pd
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
os.chdir('F:\\Gurudatt\\gg\\Topmentor\\Batch64 Day31\\Batch64 Day31\\CS 44 Ans - Model Selection and Boosting-2')
os.getcwd()
data = pd.read_csv("glass.csv")
print(data)
data.info()
data["Type"].values
types = data["Type"].values
print(np.unique(types))
fig, ax = plot.subplots()
data['Type'].value_counts().plot(ax=ax, kind='bar')


# In[4]:


#2.	Make a train_test split and fit a single decision tree classifier.
X, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
print (X)


# In[5]:


print (y)


# In[6]:


from sklearn.preprocessing import LabelEncoder
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
print (y)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)


# In[8]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[9]:


y_predict = clf.predict(X_test)


# In[10]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import  confusion_matrix
accuracy = accuracy_score(y_predict,y_test)
print(accuracy)
print(confusion_matrix(y_predict,y_test))


# In[12]:


"""3.Make a k-fold split with 3 splits and measure the accuracy score with each split 
[Hint: Refer to KFold module under sklearn’s model selection.]"""
from sklearn.model_selection import KFold
from sklearn.metrics import  confusion_matrix
k_fold = KFold(3)
print(X_train.shape,y_train.shape)


# In[13]:


models =[]
for k in enumerate(k_fold.split(X_train, y_train)):
    clf.fit(X_train,y_train)
    print(clf.score(X_test,y_test))
    models.append(clf)


# In[14]:


y_predict = clf.predict(X_test)
print ( accuracy_score(y_predict,y_test))
print(confusion_matrix(y_predict,y_test))


# In[15]:


"""4.Use gridSearchCV from sklearn for finding out a suitable number of estimators for a RandomForestClassifer 
along with a 10-fold cross validation.[Hint: Define a range of estimators and feed in range as param_grid]"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
n_estimators_range = [1,2,4,8,16,32]
grid_cv= GridSearchCV(RandomForestClassifier(),param_grid=dict(n_estimators=n_estimators_range),cv=KFold(10))
grid_cv.fit(X_train,y_train)


# In[16]:


grid_cv.best_score_


# In[17]:


grid_cv.best_estimator_


# In[18]:


y_predict = grid_cv.predict(X_test)
print ( accuracy_score(y_predict,y_test))
conf_mat = confusion_matrix(y_predict,y_test)
print(conf_mat)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




