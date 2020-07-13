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

# insights / eda
df_cleaned.columns
df_cleaned.head()
df_cleaned.info()

df_cleaned.drop_duplicates(keep=False, inplace=True)

df_cleaned.paint_color.value_counts()

df_cleaned.columns

df_cleaned.model_key.value_counts()


## feature engineering ##

# dropping not needed columns
df_cleaned = df_cleaned.drop(['maker_key', 'registration_date','feature_1','feature_2','feature_3','feature_4',
                 'feature_5','feature_6','feature_7','feature_8'], axis=1)

# seperates M-Powered cars.
df_cleaned['model_key_spaceless'] = df_cleaned['model_key']
df_cleaned.model_key_spaceless = df_cleaned.model_key_spaceless.str.replace(' ', '')
df_cleaned['model_index_inrange'] = df_cleaned['model_key_spaceless'].astype(str) + 'BMW'

df_cleaned['MPower_yn'] = df_cleaned['model_index_inrange'].apply(lambda x: 'yes' if 'm' in x[0].lower() or x[2] == 'M' else 'no')
df_cleaned.MPower_yn.value_counts()

