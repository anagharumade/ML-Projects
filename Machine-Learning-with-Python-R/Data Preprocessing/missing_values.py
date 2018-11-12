import pandas as pd

data = pd.read_csv('data.csv')
train = data.iloc[:,:-1].values
test = data.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', axis = 0, strategy = 'mean')
imputer = imputer.fit(train[:, 1:3])
train[:, 1:3] = imputer.transform(train[:, 1:3])
print(train)
