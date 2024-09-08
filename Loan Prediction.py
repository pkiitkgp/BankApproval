#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
  
data = pd.read_csv("LoanApprovalPrediction.csv") 


# In[2]:


data.head(5)


# In[9]:


obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[8]:


# Dropping Loan_ID column 
#data.drop(['Loan_ID'],axis=1,inplace=True)



# In[11]:


obj = (data.dtypes == 'object') 
object_cols = list(obj[obj].index) 
plt.figure(figsize=(18,36)) 
index = 1
  
for col in object_cols: 
  y = data[col].value_counts() 
  plt.subplot(11,4,index) 
  plt.xticks(rotation=90) 
  sns.barplot(x=list(y.index), y=y) 
  index +=1


# In[13]:


# Import label encoder 
from sklearn import preprocessing 
    
# label_encoder object knows how  
# to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
  data[col] = label_encoder.fit_transform(data[col])


# In[15]:


# To find the number of columns with  
# datatype==object 
obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))


# In[17]:


plt.figure(figsize=(12,6)) 
  
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f', 
            linewidths=2,annot=True)


# In[19]:


sns.catplot(x="Gender", y="Married", 
            hue="Loan_Status",  
            kind="bar",  
            data=data)


# In[21]:


for col in data.columns: 
  data[col] = data[col].fillna(data[col].mean())  
    
data.isna().sum()


# In[23]:


from sklearn.model_selection import train_test_split 
  
X = data.drop(['Loan_Status'],axis=1) 
Y = data['Loan_Status'] 
X.shape,Y.shape 
  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.4, 
                                                    random_state=1) 
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


# In[27]:


from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
  
from sklearn import metrics 
  
knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, 
                             criterion = 'entropy', 
                             random_state =7) 
svc = SVC() 
lc = LogisticRegression() 
  
# making predictions on the training set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_train) 
    print("Accuracy score of ", 
          clf.__class__.__name__, 
          "=",100*metrics.accuracy_score(Y_train,  
                                         Y_pred))


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Assuming X_train and Y_train are already defined and preprocessed

# Initialize the classifiers
knn = KNeighborsClassifier()
rfc = RandomForestClassifier(random_state=7)
svc = SVC(random_state=7)
lc = LogisticRegression(random_state=7)

# Define parameter grids for each classifier
param_grid_rfc = {
    'n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=10)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(10, 110, num=11)] + [None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

param_grid_knn = {
    'n_neighbors': [int(x) for x in np.linspace(start=1, stop=20, num=20)],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

param_grid_svc = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': ['scale', 'auto'],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

param_grid_lc = {
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'C': np.logspace(-4, 4, 20),
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Create RandomizedSearchCV objects for each classifier
random_search_rfc = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid_rfc, 
                                       n_iter10, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search_knn = RandomizedSearchCV(estimator=knn, param_distributions=param_grid_knn, 
                                       n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search_svc = RandomizedSearchCV(estimator=svc, param_distributions=param_grid_svc, 
                                       n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)
random_search_lc = RandomizedSearchCV(estimator=lc, param_distributions=param_grid_lc, 
                                      n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Fit the models and find the best parameters
for random_search in [random_search_rfc, random_search_knn, random_search_svc, random_search_lc]:
    random_search.fit(X_train, Y_train)
    best_model = random_search.best_estimator_
    Y_pred = best_model.predict(X_train)
    print(f"Best Accuracy score of {best_model.__class__.__name__} = {100*accuracy_score(Y_train, Y_pred):.2f}%")
    print(f"Best Parameters: {random_search.best_params_}")


# In[26]:


# making predictions on the testing set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test) 
    print("Accuracy score of ", 
          clf.__class__.__name__,"=", 
          100*metrics.accuracy_score(Y_test, 
                                     Y_pred))


# In[ ]:




