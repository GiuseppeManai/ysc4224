#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def predict(X, param):
    """Predicts the target values given the input data and the learned parameters."""
    return (X.iloc[:, 1]*param[1]) + param[0]
    

def mean_squared_error(y_pred, y_true):
    """Computes the mean squared error between the predicted and true target values."""
    return (np.sum(np.power(y_pred-y_true, 2)))/2*len(y_pred)
    
def plot_regression_line(X, y, param):
    """Plots the regression line and the scatter plot of the data."""
    plt.scatter(X.iloc[:, 1], y)
    y_pred = predict(X, param)
    plt.plot(X.iloc[:,1], y_pred, 'r')
    plt.show()


def gradient_descent(X, y, param, learning_rate, num_iterations):
    """Runs gradient descent to learn the parameters of the linear regression model."""
    m = len(X) # Number of datapoints
    cost = []
    for _ in range(num_iterations):
        loss = y - np.dot(X,param)
        grad_1 = (-2/m)*np.dot(X.iloc[:, [1]].T, loss)[0]
        grad_0 = (-2/m)*sum(loss)
        param[0] = param[0] - learning_rate * grad_0
        param[1] = param[1] - learning_rate * grad_1
        cost.append(sum(loss**2)/m)
    return param, cost


# In[3]:


get_ipython().system('jupyter nbconvert --to script LinRegrGD_template.ipynb')


# In[ ]:




