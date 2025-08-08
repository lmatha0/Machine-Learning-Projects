#Import libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split

#load dataset (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
df = pd.read_csv("diabetes.csv")
print(df.head())

#define input/output labels
X = df.drop(['Outcome'], axis=1).values
Y = df['Outcome'].values

#split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#normalize X_train
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

#define parameters
m, n= X_train.shape
W = np.zeros((n,))
b = 0.0

#calculate sigmoid for each data point
def sigmoid(X, W, b):
    z = np.dot(X, W) + b
    return 1 / (1 + np.exp(-z))

#calculate cost function (binary cross entropy) for each example
def cost(X, Y, W, b):
    m = len(X)
    err = sigmoid(X, W, b)
    eps = 1e-9 #to prevent log of 0 or 1
    return -np.mean(Y * np.log(err + eps) + (1 - Y) * np.log(1 - err + eps))

#compute gradient calculation
def gradient_calculation(X, Y, W, b):
    m = len(X)
    e = sigmoid(X, W, b)
    error = e - Y
    dj_dw = (1/m) * np.dot(X.T, error)
    dj_db = (1/m) * np.sum(error)
    return dj_dw, dj_db

def gradient_descent(X, Y, W, b, cost, gradient_calculation, alpha, iters):
    loss_values = []

    for i in range(iters):
        dj_dw, dj_db = gradient_calculation(X, Y, W, b)
        loss_values.append(cost(X, Y, W, b))
        W -= alpha * dj_dw
        b -= alpha * dj_db

        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {loss_values[-1]}")

    return W, b, loss_values

W_final, b_final, J_values = gradient_descent(X_train, Y_train, W, b, cost, gradient_calculation, 0.5, 100)

#return 1 if predicted value >0.5
def predict(X, W, b):
    return (sigmoid(X, W, b) >= 0.5).astype(int)

#calculate test accuracy
Y_pred = predict(X_test, W_final, b_final)
accuracy = np.mean(Y_pred == Y_test) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

