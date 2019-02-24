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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0 )
