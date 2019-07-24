#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# In[2]:


data = pd.read_csv("data.csv")
sample = pd.read_csv("sample_submission.csv")


# In[3]:


X = data[['remaining_min', 'power_of_shot', 'knockout_match', 'game_season',
       'remaining_sec', 'distance_of_shot', 'area_of_shot',
       'shot_basics', 'range_of_shot',
       'home/away','shot_id_number', 'lat/lng', 'type_of_shot',
       'type_of_combined_shot', 'remaining_min.1',
       'power_of_shot.1', 'knockout_match.1', 'remaining_sec.1',
       'distance_of_shot.1']]


# In[4]:


y = data['is_goal']


# In[5]:


X_train_df = X[y.notnull()]
y_train_df = y[y.notnull()]
X_test_df = X[~y.notnull()]
y_test_df = y[~y.notnull()]
s = sample["shot_id_number"].values -1
y_test_df = y_test_df[s]
X_test_df = X_test_df.loc[y_test_df.index]


# In[6]:


le = LabelEncoder()


# In[7]:


sel = ['remaining_min', 'power_of_shot', 'knockout_match',
       'remaining_sec', 'distance_of_shot', 'remaining_min.1',
       'power_of_shot.1','shot_id_number', 'knockout_match.1', 'remaining_sec.1',
       'distance_of_shot.1']

for i in sel :
    X_train_df[i].fillna(np.mean(X_train_df[i]), inplace=True)
    X_test_df[i].fillna(np.mean(X_test_df[i]), inplace=True)
    


# In[8]:


sel = ['game_season', 'area_of_shot',
       'shot_basics', 'range_of_shot',
       'home/away', 'lat/lng', 'type_of_shot',
       'type_of_combined_shot']
for i in sel :
    X_train_df[i].fillna("", inplace=True)
    X_test_df[i].fillna("", inplace=True)


# In[9]:


sel = ['game_season', 'area_of_shot',
       'shot_basics', 'range_of_shot',
       'home/away', 'lat/lng', 'type_of_shot',
       'type_of_combined_shot']
for i in sel :
    X_train_df[i] = le.fit_transform(X_train_df[i])
    X_test_df[i] = le.fit_transform(X_test_df[i])



# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[11]:


model = DecisionTreeClassifier(max_depth=50, criterion="entropy")


# In[12]:


model.fit(X_train_df, y_train_df)


# In[13]:


y_result = model.predict(X_test_df)


# In[14]:


d = {
    "shot_id_number" : s+1,
    "is_goal" : y_result
}


# In[15]:


df1 = pd.DataFrame(d)


# In[16]:


df1.to_csv("output.csv",index=False)


# In[ ]:




