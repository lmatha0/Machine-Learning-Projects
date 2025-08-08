This is a repository for various machine learning projects to understand the code behind algorithms. The idea is to gain more familiarity by implementing these algorithms from scratch to develop stronger coding skills and understand the mathematical concepts behind the algorithms used.

Projects:

linear_reg:
- This is used to understand the preparation of data and calculations involved in regression.
- Loads a CSV file, normalizes data, calculates cost function, calculates gradient descent

car_price_prediction:
- A multivariate linear regression project that predicts car prices based on features including km_driven, transmission, # of owners, and more
- Uses panda to load and clean up data, scikitlearn to normalize data, numpy to calculate cost function and gradient descent, and matplotlib to plot optimized parameters
- Also evaluates R2 score manually and implements LinearRegression() and metrics from scikitlearn to compare R2 score valuess

diabetes_prediction:
- A multivariate logistic regression project that predicts the probability of heart disease based on factors including BMI, blood pressure, insulin levels and more
- Uses panda to load and clean data, numpy to vectorize calculations, and scikitlearn to split data into test and train sets
- computes accuracy as a percentage after training the model
