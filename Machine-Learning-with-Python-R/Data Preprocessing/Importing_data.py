import pandas as pd

data = pd.read_csv('Data.csv')
train_ind = data.iloc[:, :-1]
countries = data.to_csv(columns = ['Country'], index = False) #Creates a string. can be split by spaces to obtain words
countries = data[['Country','Age']] # Creating a data frame using column headers
print(countries)
