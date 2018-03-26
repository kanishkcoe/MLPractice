# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:55:26 2018

@author: kanis
"""

import numpy as np
import pandas as pd

training_data = pd.read_csv("dbTrain.csv")

x_train = training_data.iloc[:, :-1].values
y_train = training_data.iloc[:, -1].values

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

#predicting results

test_data = pd.read_csv("dbTest.csv")

predictions = regressor.predict(test_data)

import csv

with open("dbResult.csv", "w", newline='') as f:
  theWriter = csv.writer(f)
  li = predictions
  li = li.reshape((111, 1))  
  
  for i in range(len(li)):
    theWriter.writerow(li[i])
  