#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 07:52:48 2020

@author: Mario
"""


import pandas as pd

# all columns are displayed
pd.options.display.max_columns = None
pd.options.display.max_rows = None

# imports dataset
df = pd.read_csv("bmw_pricing_challenge.csv")

df_cleaned = df

df_cleaned.columns
df_cleaned.head()
df_cleaned.info()

df_cleaned.drop_duplicates(keep=False, inplace=True)

df_cleaned.paint_color.value_counts()

df_cleaned = df_cleaned.drop(['maker_key', 'registration_date','feature_1','feature_2','feature_3','feature_4',
                 'feature_5','feature_6','feature_7','feature_8'], axis=1)

df_cleaned.columns

df_cleaned.model_key.value_counts()

## feature engineering ##

# seperates most M-Powered cars.
df_cleaned['MPower_yn'] = df_cleaned['model_key'].apply(lambda x: 'yes' if 'm' in x[0].lower() else 'no')
df_cleaned.MPower_yn.value_counts()

# seperates M-Powered X series cars and merges back into main dataframe
df_cleaned['X_series'] = df_cleaned['model_key'].apply(lambda x: 'yes' if 'x' in x[0].lower() else 'no')
df_XSeries = df_cleaned[(df_cleaned['X_series'] == "yes")]
df_XSeries['MPower_Xseries_yn'] = df_XSeries.model_key.str[3]
df_cleaned['MPower_Xseries_yn']= df_XSeries['MPower_Xseries_yn']

df_cleaned['MPower_yn'] = df_cleaned['MPower_Xseries_yn'].apply(lambda x: 'yes' if 'm' in x.lower() else 'no')
