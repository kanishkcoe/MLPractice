# -*- coding: utf-8 -*-

#gradient descent for multiple features
#importing libraries
import numpy as np
import pandas as pd

#retrieving data from the csv file
training_data = pd.read_csv("gdtrain.csv")
x_train = training_data.iloc[:, :-1]
y_train = training_data.iloc[:, -1]

test_data = pd.read_csv("gdtest.csv")
x_test = test_data


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.transform(x_test)

x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

x_train['ones'] = np.ones(379)
x_test['ones'] = np.ones(127)

x = np.array(x_train)
y = np.array(y_train)

#hyper parameters
learning_rate = 0.0001

#initial coefficients 
m = np.zeros(14)

#number of iterations
num_iterations = 10000

#compute error for a given point
def compute_error_for_given_point(m, x, y):
  totalError = 0
  
  for i in range(0, len(x)):
    totalError += (y[i] - sum(m * x[i])) ** 2
  
  return totalError / float(len(x))


#step gradient function
def step_gradient(current_m, x, y, learning_rate):
  m_gradient = np.zeros(len(current_m))
  n = float(len(x))
  
  for i in range(len(x)):
    m_gradient += (-2 / n) * x[i] * (y[i] - sum(current_m * x[i]))
  new_m = current_m - (learning_rate * m_gradient)
  
  return new_m
    

#gradient runner
def gradient_descent_runner(x, y, start_m, learning_rate, num_iterations):
  m = start_m
  
  for i in range(num_iterations):
    m = step_gradient(m, x, y, learning_rate)
  return m

result_m = gradient_descent_runner(x, y, m, learning_rate, num_iterations)

x_test = np.array(x_test)

def predict(data, m):
  result = np.zeros(len(data))
  for i in range(len(data)):
    result[i] = sum(data[i] * m)
  return result
  
test_prediction = predict(x_test, result_m)

#exporting result
import csv

with open("gdResult.csv", "w", newline='') as f:
  theWriter = csv.writer(f)
  li = test_prediction
  li = li.reshape((127, 1))  
  
  for i in range(len(li)):
    theWriter.writerow(li[i])
  