#Import libraries
import numpy as np #numpy to vectorize calculations
import matplotlib.pyplot as plt #matplotlib to plot data between input/output variables
import pandas as pd #panda to load csv files and format them for numpy calculations
from sklearn.model_selection import train_test_split #split data into train and test sets
from sklearn.preprocessing import StandardScaler #scale input features

#Loading the dataset using pandas to format the data
df = pd.read_csv('car data.csv')

#Printing the first 5 rows of the dataset
print("Initial Data")
print(df.head())

#Clean up data, remove Car_Name as data cannot be processed during regression. Calculate Car_Age using Year of the car
df['Current_Year'] = 2025
df['Car_Age'] = df['Current_Year'] - df['Year']
df = df.drop(['Car_Name', 'Year', 'Current_Year'], axis=1)

df = pd.get_dummies(df, drop_first=True) #Encode categorical variables as True/False

print("Cleaned Data")
print(df.head()) #Print the cleaned up data

#Set up independant and dependant variables
X = df.drop(['Selling_Price'], axis=1).values #Input of features which is an array with shape (m, n)
Y = df['Selling_Price'].values.reshape(-1, 1) #Output of Selling_Price reshaped to match X shape of m rows

#split data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#scale values of X by X - mean(X) / std(X)
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Initialize parameters W and b with m examples and n features
m = X_train.shape[0]
n = X_train.shape[1]
W = np.zeros((n,))
b = 0

#predict calculation to calculate the predicted y value based on the input parameter x and its parameters w,b
def predict(X, W, b):
    p = np.dot(X, W.reshape(-1, 1)) + b
    return p

#cost function implementation to calculate error value between predicted y value and actual y value
def cost(X, Y, W, b):
    m = X.shape[0]
    cost = 0.0 #initialize cost to 0.0 and update over examples

    for i in range(m):
        p = predict(X[i], W, b)
        cost += (p - Y[i][0])**2
    cost = cost / (2 * m) 
    return cost 

#gradient descent calculation to calculate update value parameters of w and b
    #two for loops, one for m examples, second for n features in each m row
def gradient_calculation(X, Y, W, b):
    m, n = X.shape
    dj_dw = np.zeros((n,)) #initialize dj_dw as an array of 0s based on number of features to match dimension for vectorization
    dj_db = 0.0 #dj_db is a vector 

    for i in range(m):
        err_example = (np.dot(X[i], W) + b) - Y[i][0]
        for j in range(n):
            dj_dw[j] += err_example * X[i][j] 
        dj_db += err_example 
    dj_db /= m #gradient descent formula for dj_db
    dj_dw /= m #gradient descent formula for dj_dw

    return dj_dw, dj_db
        
#gradient descent implentation to update parameters using calculated values over iterations
def gradient_descent(X, Y, W, b, cost, gradient_calculation, alpha, iters):
    J_values = [] #store cost function values over iterations to see optimization

    for i in range(iters):
        dj_dw, dj_db = gradient_calculation(X, Y, W, b) #perform gradient descent at each step

        W -= alpha * dj_dw #updating dj_dw over each iteration
        b -= alpha * dj_db #updating dj_db over each iteration

        J_values.append(cost(X, Y, W, b)) #adding cost value to J_values list

        #Viewing the iteration count and cost at specified iteration
        if i % 10 == 0:
            print(f"Iteration {i}: Cost = {J_values[-1]}")

    return W, b, J_values  

#Store optimized values in a variable
W_final, b_final, J_values = gradient_descent(X_train, Y_train, W, b, cost, gradient_calculation, 0.05, 300) #alpha and number of iterations chosen arbitrarily
print("W mean:", np.mean(W_final), "W std:", np.std(W_final))
print("b mean:", np.mean(b_final))

#plotting cost function over iterations
plt.plot(range(len(J_values)), J_values)
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost over Iterations")
plt.show()

# Plot actual vs predicted
Y_pred_test = predict(X_test, W_final, b_final)
plt.scatter(Y_test, Y_pred_test)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Test Set: Actual vs Predicted Selling Prices")
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red')
plt.show()

#evaluating data using r2 score
def r2_score(y_true, y_pred):
    residual = np.sum((y_true - y_pred) ** 2)
    total = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (residual / total)
    return r2

r2 = r2_score(Y_test, Y_pred_test)
print(f"R2 Score: {r2}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score as sk_r2

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred_sklearn = model.predict(X_test)
print("Sklearn R²:", sk_r2(Y_test, Y_pred_sklearn))