#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 08:58:48 2020

@author: Mario
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
pd.options.display.max_rows = None

df = pd.read_csv('bmw_cleaned.csv')

# choose revelant columns
df.columns

df_model = df[['model_key', 'mileage', 'fuel', 'paint_color', 'car_type', 'price',
       'sold_at', 'MPower_yn', 'lowMiles_yn', 'highMiles_yn',
       'superHighMiles_yn', 'fast_yn', 'condition', 'sold_winter',
       'sold_spring', 'sold_summer', 'sold_fall']]

# get dummy data / creates categorical data into integers
df_dum = pd.get_dummies(df_model)

# train test split (80/20)
from sklearn.model_selection import train_test_split

X = df_dum.drop('price', axis = 1)
y = df_dum.price.values # creates an array - recommended to use for models

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 1)

# single linear regression (MAE = 3161.63)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

lm = LinearRegression()

# train the model
lm.fit(X_train, y_train)

# preform prediction on the test data
y_pred = lm.predict(X_test)

# performance metrics
print('Coefficients:', lm.coef_)
print('Intercept: ', lm.intercept_)
print('Mean absolute error (MAE): %.2f' % mean_absolute_error(y_test, y_pred))


# 10 Fold Cross Validation (to generalize data) (MAE = -3206.52)
from sklearn.model_selection import cross_val_score

np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# lasso regression  (L1 Regularization) (MAE = -3192.81)
from sklearn.linear_model import Lasso

lm_las = Lasso() # alpha defaults to 1
lm_las.fit(X_train, y_train)
np.mean(cross_val_score(lm_las, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))

# Find the optimal alpha

alpha = []
err = []

for i in range(1, 100):
  alpha.append(i/100)
  lmlas = Lasso(alpha = (i/100))
  err.append(np.mean(cross_val_score(lmlas, X_train, y_train, scoring = 'neg_mean_absolute_error', cv = 10)))

plt.plot(alpha,err)

# shows that alpha is best at 0.99 (shows where optimal alpha is)
err = tuple(zip(alpha,err))
df_err = pd.DataFrame(err, columns = ['alpha','err'])
df_err[df_err.err == max(df_err.err)]

lm_las = Lasso(0.99) # alpha defaults to 1
lm_las.fit(X_train, y_train)
np.mean(cross_val_score(lm_las, X_train, y_train, scoring = 'neg_mean_absolute_error', cv= 10))