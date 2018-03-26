#multiple linear regression
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('gdtrain.csv')
x_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values

#fit the multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#testing the performance of the regressor on test set
#predicting the results
#predicting the results in test cases
x_test = pd.read_csv("gdtest.csv")

y_prediction = regressor.predict(x_test)

import csv

with open("gdResult.csv", "w", newline='') as f:
  theWriter = csv.writer(f)
  li = y_prediction
  li = li.reshape((127, 1))  
  
  for i in range(len(li)):
    theWriter.writerow(li[i])

