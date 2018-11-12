# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:47:49 2017

@author: absol
"""

import pandas as pd

data = pd.read_csv('data.csv')
X = data.iloc[:,:-1].values
Y = data.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_test = LabelEncoder()
Y = labelencoder_test.fit_transform(Y)