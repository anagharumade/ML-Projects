# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 13:53:11 2017

@author: absol
"""

import pandas as pd

data = pd.read_csv('data.csv')
#Splitting the independent and the dependent variables
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values

#Dealing with Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Converting Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)

#Splitting into training and test
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)