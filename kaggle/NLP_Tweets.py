# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:02:16 2020

@author: Orange
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('train.csv')

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
### text clean 
data = []
for i in range(0,7613):
    review = re.sub("[^a-zA-Z]"," ",dataset["text"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    data.append(review)
### keyword clean
#import re
#import nltk
#nltk.download('stopwords')
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#data_keyword = []
#for i in range(0,7613):
#    review = re.sub("[^a-zA-Z]"," ",dataset["keyword"][i])
#    review = review.lower()
#    review = review.split()
#    ps = PorterStemmer()
#    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
#    review = " ".join(review)
#    data_keyword.append(review)    
    
### location 
# Encoding the Independent Variable
#X = dataset.values 
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X = LabelEncoder()
#X[:,3] = labelencoder_X.fit_transform(X[:,3])
#onehotencoder = OneHotEncoder(categorical_features = 'all')
#X = onehotencoder.fit_transform(X[:,3]).toarray()


   
from sklearn.feature_extraction.text import CountVectorizer
cv =  CountVectorizer(max_features = 2000)
x = cv.fit_transform(data).toarray()
y = dataset.iloc[:,4].values
### randomForestClassifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 500, criterion = 'entropy', random_state = 0 )
classifier.fit(x, y)
### naive_bayes
from sklearn.naive_bayes import GaussianNB
classifier_naive = GaussianNB()
classifier_naive.fit(x, y)
### logisticRegression
from sklearn.linear_model import LogisticRegression
classifier_logistic = LogisticRegression(random_state = 0)
classifier_logistic.fit(x, y)



dataset_test = pd.read_csv('test.csv')
data_test = []
for i in range(0,3263):
    review = re.sub("[^a-zA-Z]"," ",dataset_test["text"][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = " ".join(review)
    data_test.append(review)
cv =  CountVectorizer(max_features = 2000)
x_test = cv.fit_transform(data_test).toarray()

y_pred = classifier.predict(x_test)
y_pred_naive = classifier_naive.predict(x_test)
y_pred_logistic = classifier_logistic.predict(x_test)
answer = pd.read_csv('sample_submission.csv')
y_test = answer.iloc[:,1].values
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm_naive = confusion_matrix(y_test, y_pred_naive)
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
