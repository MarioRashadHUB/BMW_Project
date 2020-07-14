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

# determine if car is high milage or low milage
df_cleaned = df_cleaned[df_cleaned['mileage'] > 0]
df_cleaned['lowMiles_yn'] = df_cleaned['mileage'].apply(lambda x: 'yes' if x < 99000 else 'no') 
df_cleaned['highMiles_yn'] = df_cleaned['mileage'].apply(lambda x: 'yes' if x > 150000 else 'no')
df_cleaned['superHighMiles_yn'] = df_cleaned['mileage'].apply(lambda x: 'yes' if x > 300000 else 'no')

# determine if car is fast or not  
# After analyzing the engine_power column, I realized that the column for the most part is very inaccurate and cannot be used.
df_cleaned = df_cleaned[df_cleaned['engine_power'] > 0]
df_cleaned['fast_yn'] = df_cleaned['engine_power'].apply(lambda x: 'yes' if x > 250 else 'no')
df_cleaned = df_cleaned.drop(['engine_power'], axis=1)


# determine if car is new or used
df_cleaned['condition'] = df_cleaned['mileage'].apply(lambda x: 'new' if x < 50000 else 'used') 

# dropping cars that have been sold for less then $1000.  Possibly had mechinical issues and can be outliers in dataset.
df_cleaned = df_cleaned[df_cleaned['price'] > 1000]

# export cleaned dataframe as csv
df_cleaned.to_csv('bmw_cleaned.csv',index = False)

