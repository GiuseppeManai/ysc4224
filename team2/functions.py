import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

def mean_squared_error(y_true, y_predicted):
    # Calculating the loss or cost
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost
 
def predict(X, params):
    """Predicts the target values given the input data and the learned parameters."""
    weight, bias = params
    y_pred = (X * weight) + bias
    return y_pred


def plot_cost_function(weights, costs) :
    """Plots the cost function after finding params using gradient descent"""
    # Visualizing the weights and cost at for all iterations
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()


def plot_regression_line(X, y, params) :
    """Plots the regression line and the scatter plot of the data."""
    y_pred = predict(X, params)
    plt.scatter(X, y, marker='o', color='red')
    plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='blue',markerfacecolor='red',
             markersize=10,linestyle='dashed')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def gradient_descent(x, y, params, learning_rate, num_iterations):    
    """Runs gradient descent to learn the parameters 
    of the linear regression model."""

    # Initializing weight, bias and stopping threshold
    stopping_threshold = 1e-6
    current_weight, current_bias = params
    n = float(len(x))
     
    costs = []
    weights = []
    previous_cost = None
     
    # Estimation of optimal parameters
    for i in range(num_iterations):
        
        # Making predictions
        y_predicted = predict(x, (current_weight, current_bias))
         
        # Calculating the current cost
        current_cost = mean_squared_error(y, y_predicted)
 
        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
 
        costs.append(current_cost)
        weights.append(current_weight)
         
        # Calculating the gradients
        weight_derivative = -(2/n) * sum(x * (y-y_predicted))
        bias_derivative = -(2/n) * sum(y-y_predicted)
         
        # Updating weights and bias
        current_weight = current_weight - (learning_rate * weight_derivative)
        current_bias = current_bias - (learning_rate * bias_derivative)
                 
     
    return {"fitted_weight" : current_weight, 
    "fitted_bias" : current_bias,
    "weights" : weights, 
    "costs" : costs}