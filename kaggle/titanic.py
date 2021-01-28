# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 19:10:03 2020

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
dataset = pd.read_csv("train.csv")
df = dataset
# find the missing data
df.isnull().sum()   # the age, embarked and Cabin have losting data
# pick Survived list 
y = dataset.iloc[:,1].values
# fill in the data
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Age']  = df['Age'].fillna(df['Age'].median())
# alter the sex from str to int 0,1
df['Sex'] = df['Sex'].map({'male': 0,'female': 1})
# alter the Embarked str to int 0 1 2
df['Embarked'] = df['Embarked'].map({'S': 0,'C': 1,'Q':2})
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].median())

df['Cabin'] = df['Cabin'].fillna('M')
import re
data = []
for i in range(0,891):
    review = re.sub("[^a-zA-Z]"," ",df["Cabin"][i][0])
    data.append(review)
data = pd.DataFrame(data)
df['Deck'] = data
df['Deck'] = df['Deck'].map({'A': 1,'B': 2,'C':3, 'D':4, 'E':5, 'F':6,'G':7,'T':8,'M':0})
df['FamilySize'] = df ['SibSp'] + df['Parch'] + 1

# plot the figure SibSp Fare Embarked
plt.subplots(figsize=(16,14))
plt.subplot(231)
plt.hist(x = [df[df['Survived']==1]['SibSp'], df[df['Survived']==0]['SibSp']], 
         stacked=True, color = ['g','r'],label = ['Survived','Dead'])
plt.xlabel('SibSp')
plt.legend()

plt.subplot(232)
plt.hist(x=[df[df['Survived']==1]['Fare'],df[df['Survived']==0]['Fare']], stacked = True, color = ['g','r'], label = ['Survived','Dead'])
plt.legend()
plt.xlabel('Fare')

plt.subplot(233)
plt.hist(x=[df[df['Survived']==1]['Pclass'],df[df['Survived']==0]['Pclass']], stacked = True, color = ['g','r'], label = ['Survived','Dead'])
plt.legend()
plt.xlabel('Pclass')

# sns plot the figure 
fig, qaxis = plt.subplots(1,3,figsize=(14,12))
sns.barplot(x='Sex',y='Survived',hue="Pclass",data = df, ax=qaxis[0])
sns.barplot(x='Sex',y='Survived',hue="Parch",data = df, ax=qaxis[1])
sns.barplot(x='Sex',y='Survived',hue="Embarked",data = df, ax=qaxis[2])

# pick the important data 
df_1=df[['Pclass', 'Sex', 'Age', 'Fare','Embarked', 'Deck', 'FamilySize']].copy()
df_1 = df_1.values

# change the scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_1 = sc.fit_transform(df_1)

# ANN learning
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#
classifier = Sequential()
#
## Adding the input layer and the first hidden layer
classifier.add(Dense(units = 4,  activation = "relu", kernel_initializer = "uniform", input_dim = 7))
#
## Adding second hidden layer
classifier.add(Dense(units = 4,  activation = "relu", kernel_initializer = "uniform"))
## Adding second hidden layer
classifier.add(Dense(units = 3,  activation = "relu", kernel_initializer = "uniform"))
## Adding output layer
classifier.add(Dense(units = 1,  activation = "sigmoid", kernel_initializer = "uniform"))
#
## if the output not only so change to activation = 'softmax
##compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
r = classifier.fit(df_1, y, batch_size = 800, epochs = 1000)

### XGBOOST

from xgboost import XGBClassifier
classifier_xgboost = XGBClassifier()
classifier_xgboost.fit(df_1, y)
### score of xgboost
accuracies = []
from sklearn.model_selection import cross_val_score
xgboost_accuracies = cross_val_score(estimator = classifier_xgboost, X = df_1, y = y, cv = 10)
xgboost = xgboost_accuracies.mean()
xgboost_accuracies.std()
accuracies.append(xgboost)


### RandomForest
from sklearn.ensemble import RandomForestClassifier
classifier_RandomForest = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0 )
classifier_RandomForest.fit(df_1, y)

### score of RandomForest
RandomForest_accuracies = cross_val_score(estimator = classifier_RandomForest, X = df_1, y = y, cv = 10)
RandomForest = RandomForest_accuracies.mean()
RandomForest_accuracies.std()
accuracies.append(RandomForest)

# test processing 
dataset_test = pd.read_csv("test.csv")
dataset_answer = pd.read_csv("gender_submission.csv")
y_test = dataset_answer.iloc[:,1].values
df_test = dataset_test
df_test.isnull().sum()
df_test['Fare'] = df_test['Fare'].fillna(df_test['Fare'].median())
df_test['Age']  = df_test['Age'].fillna(df_test['Age'].median())
df_test['Sex'] = df_test['Sex'].map({'male': 0,'female': 1})
df_test['Embarked'] = df_test['Embarked'].map({'S': 0,'C': 1,'Q':2})
df_test['Embarked'] = df_test['Embarked'].fillna(df_test['Embarked'].median())
df_test['Cabin'] = df_test['Cabin'].fillna('M')
data = []
for i in range(0,891):
    review = re.sub("[^a-zA-Z]"," ",df["Cabin"][i][0])
    data.append(review)
data = pd.DataFrame(data)
df_test['Deck'] = data
df_test['Deck'] = df_test['Deck'].map({'A': 1,'B': 2,'C':3, 'D':4, 'E':5, 'F':6,'G':7,'T':8,'M':0})
df_test['FamilySize'] = df_test['SibSp'] + df_test['Parch'] + 1
df_2=df_test[['Pclass', 'Sex', 'Age', 'Fare','Embarked', 'Deck', 'FamilySize']].copy()
df_2 = df_2.values
df_2 = sc.transform(df_2)

y_pred = r.model.predict(df_2)
y_pred_xgboost = classifier_xgboost.predict(df_2)
y_pred_RandomForest = classifier_RandomForest.predict(df_2)
new_y_pred = []
for var in y_pred:
    if var >0.5:
        new_y_pred.append(1)
    else:
        new_y_pred.append(0)
# confusion matrix to answer
from sklearn.metrics import confusion_matrix
y_pred = y_pred > 0.5
cm_xgboost = confusion_matrix(y_test, y_pred_xgboost)
cm_RandomForest = confusion_matrix(y_test, y_pred_RandomForest) 
cm_ANN = confusion_matrix(y_test, y_pred)

#from sklearn.model_selection import cross_val_score
#accuracies_ANN = cross_val_score(estimator = classifier, X = df_1, y = y, cv = 10)
#ANN = accuracies_ANN.mean()
#accuracies_ANN.std()
#accuracies.append(ANN)

#scores = cross_val_score(kNeighborsClassifier, X, Y, cv=5)
#scores.mean()

#Backward Elimination
import statsmodels.api as sm
df_1 = np.append(arr = np.ones((891, 1)).astype(int), values = df_1, axis = 1)
X_opt = df_1 [:, [0, 1, 2, 3, 4, 5, 6]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

### fare isn't important
X_opt = df_1 [:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#from sklearn.decomposition import PCA
#pca = PCA(n_components = "None")
#X_train=df[['Pclass', 'Sex', 'Age', 'Fare','Embarked', 'Deck', 'FamilySize']].copy()
##X_train = sc.fit_transform(X_train) 
#X_test = df_test[['Pclass', 'Sex', 'Age', 'Fare','Embarked', 'Deck', 'FamilySize']].copy()
#X_test  = sc.fit_transform(X_test ) 
##X_train = pca.fit_transform(X_train)
#X_test = pca.transform(X_test)
####來看比率的
#explained_variance = pca.explained_variance_ratio_


# save to csv
result = pd.DataFrame(data = {'PassengerId':df_test['PassengerId'], 'Survived':new_y_pred})
result.to_csv("Submission.csv", index=False)

