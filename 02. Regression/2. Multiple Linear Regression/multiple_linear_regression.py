# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 01:08:52 2019

@author: Saurabh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('50_Startups.csv');
print(dataset.head())
X = dataset.iloc[:, :-1].values #exclusde the last column
y = dataset.iloc[: , 4].values #just the 4th column


# encoding the categorical data as dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[: , 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # can drop any one among all the dummy variables, so as to avoid D' = 1 - D

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#fitting the multiple linear regresison model to traningn set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting
y_pred = regressor.predict(X_test)



import statsmodels.formula.api as sm
X = np.append(np.ones((50, 1)).astype(int), values = X, axis = 1)

X_opt = X[:, [0, 1, 2, 3, 4, 5]] # include all wors and 0-5 columns
# Here endog is the dependent variable, and exog are the independent variables
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit() # more : https://en.wikipedia.org/wiki/Ordinary_least_squares
regressor_ols.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5]] # include all wors and 0-5 columns
# Here endog is the dependent variable, and exog are the independent variables
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit() # more : https://en.wikipedia.org/wiki/Ordinary_least_squares
regressor_ols.summary()