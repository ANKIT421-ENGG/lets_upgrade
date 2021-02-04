#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("https://raw.githubusercontent.com/ingledarshan/AIML-B2/main/data.csv")


# In[4]:


df.head()


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df['Unnamed: 32']


# In[9]:


df = df.drop("Unnamed: 32", axis=1)


# In[10]:


df.head()


# In[11]:


df.columns


# In[12]:


df.drop('id', axis=1, inplace=True)


# In[13]:


df.columns


# In[14]:


type(df.columns)


# In[15]:


l = list(df.columns)
print(l)


# In[16]:


features_mean = l[1:11]
features_se = l[11:21]
features_worst = l[21:]


# In[17]:


print(features_mean)


# In[18]:


print(features_se)


# In[19]:


print(features_worst)


# In[21]:


df['diagnosis'].unique()


# In[22]:


sns.countplot(df['diagnosis'], label="Count",);


# In[23]:


df['diagnosis'].value_counts()


# In[24]:


df.shape


# In[25]:


df.describe()


# In[26]:


len(df.columns)


# In[27]:


corr = df.corr()
corr


# In[28]:


corr.shape


# In[29]:


plt.figure(figsize=(8,8))
sns.heatmap(corr);


# In[30]:


df.head()


# In[31]:


df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0})


# In[32]:


df.head()


# In[33]:


df['diagnosis'].unique()


# In[34]:


X = df.drop('diagnosis', axis=1)
X.head()


# In[35]:


y = df['diagnosis']
y.head()


# In[36]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[37]:


df.shape


# In[38]:


X_train.shape


# In[39]:


X_test.shape


# In[40]:


y_train.shape


# In[43]:


y_test.shape


# In[44]:


X_train.head(1)


# In[45]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)


# In[46]:


X_train


# # Logistic Regression

# In[47]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[48]:


y_pred = lr.predict(X_test)


# In[49]:


y_pred


# In[50]:


y_test


# In[51]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[52]:


lr_acc = accuracy_score(y_test, y_pred)
print(lr_acc)


# In[53]:


results = pd.DataFrame()
results


# In[55]:


tempResults = pd.DataFrame({'Algorithm':['Logistic Regression Method'], 'Accuracy':[lr_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Decision Tree 

# In[57]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


# In[58]:


y_pred = dtc.predict(X_test)
y_pred


# In[59]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[60]:


dtc_acc = accuracy_score(y_test, y_pred)
print(dtc_acc)


# In[61]:


tempResults = pd.DataFrame({'Algorithm':['Decision tree Classifier Method'], 'Accuracy':[dtc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Random Forest

# In[62]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


# In[63]:


y_pred = rfc.predict(X_test)
y_pred


# In[64]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[65]:


rfc_acc = accuracy_score(y_test, y_pred)
print(rfc_acc)


# In[66]:


tempResults = pd.DataFrame({'Algorithm':['Random Forest Classifier Method'], 'Accuracy':[rfc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results


# # Support Vector 

# In[67]:


from sklearn import svm
svc = svm.SVC()
svc.fit(X_train,y_train)


# In[68]:


y_pred = svc.predict(X_test)
y_pred


# In[69]:


from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[70]:


svc_acc = accuracy_score(y_test, y_pred)
print(svc_acc)


# In[71]:


tempResults = pd.DataFrame({'Algorithm':['Support Vector Classifier Method'], 'Accuracy':[svc_acc]})
results = pd.concat( [results, tempResults] )
results = results[['Algorithm','Accuracy']]
results

