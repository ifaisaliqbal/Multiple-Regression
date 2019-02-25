# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 10:39:55 2019

@author: iqbal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("50_Startups.csv");

X = data.iloc[:,:-1].values;
y = data.iloc[:,4].values;

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])

oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0 )

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm

X_opt = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

import statmodels.formula.api as sm

regressor_OLS = sm.OLS(endog = y, exog = X_opt)
