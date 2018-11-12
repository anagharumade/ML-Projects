# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 17:06:08 2017

@author: absol
"""

import pandas as pd

#importing dataset
data = pd.read_csv('data.csv')

#Splitting dataset into Dependent and independent variables
X = data.iloc[:, :-1].values
Y = data.iloc[:, 3].values

#Dealing with missing values
from sklearn.preprocessing import Imputer
imputer  = Imputer(missing_values = 'NaN', axis = 0, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder = LabelEncoder()
Y = labelencoder.fit_transform(Y)

#Splitting into training and testing data
from sklearn.cross_validation import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size = 0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
Y_train = scaler_X.transform(Y_train)