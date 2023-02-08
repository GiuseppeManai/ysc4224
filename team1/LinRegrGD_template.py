#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def predict(X, param):
    """Predicts the target values given the input data and the learned parameters."""
    return param[0] + param[1] * X

def mean_squared_error(y_pred, y_true):
    """Computes the mean squared error between the predicted and true target values."""
    m = len(y_true)
    return (1 / m) * np.sum((y_pred - y_true)**2)
    
def plot_regression_line(X, y, param):
    """Plots the regression line and the scatter plot of the data."""
    plt.scatter(X, y, s=10, alpha=0.8)
    y_pred = param[0] + param[1] * X
    plt.plot(X, y_pred, color='r', label='Predicted line: {} + {}x'.format(round(param[0], 2), round(param[1], 2)))
    
def gradient_descent(X, y, param, learning_rate, num_iterations):
    """Runs gradient descent to learn the parameters of the linear regression model."""

    m = y.shape[0] # number of samples
    costs = []
    for i in range(num_iterations):
        y_pred = np.dot(X, param)
        loss = y_pred - y
        gradient = (2 / m) * np.dot(X.T, loss)
        param = param - learning_rate * gradient
        
        J = mean_squared_error(y_pred, y)

        costs.append(J)
        # print("Iteration %d | Cost: %f" % (i, J))
        
    return param, costs


# In[3]:


get_ipython().system('jupyter nbconvert --to script LinRegrGD_template.ipynb')

