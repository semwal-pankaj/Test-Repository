#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 13:03:37 2020

@author: pankajsemwal
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


# Defination of Imputer functionality
def Imputer(X_train,X_test):
    # differentiate between categorical/numerical columns
    catcol=X_train.dtypes=='object'
    categoricalcol=catcol[catcol].index
    numericalcol=catcol[~catcol].index
    
    #Imputation of categorical columns with most frequent value
    CatImputer=SimpleImputer(strategy="most_frequent")
    X_traincatimp=pd.DataFrame(CatImputer.fit_transform(X_train[categoricalcol]))
    X_testcatimp=pd.DataFrame(CatImputer.transform(X_test[categoricalcol]))
    X_traincatimp.columns=X_train[categoricalcol].columns
    X_testcatimp.columns=X_test[categoricalcol].columns
    
    #Imputation of numerical columns with mean value
    NumImputer=SimpleImputer(strategy="mean")
    X_trainnumimp=pd.DataFrame(NumImputer.fit_transform(X_train[numericalcol]))
    X_testnumimp=pd.DataFrame(NumImputer.fit_transform(X_test[numericalcol]))
    X_trainnumimp.columns=X_train[numericalcol].columns
    X_testnumimp.columns=X_test[numericalcol].columns

    #Merge the categorical columns and numerical columns to get back the complete data
    X_trainpostimp=pd.concat([X_traincatimp,X_trainnumimp],axis=1)
    X_testpostimp=pd.concat([X_testcatimp,X_testnumimp],axis=1)
    
    return X_trainpostimp, X_testpostimp


# Defination of label encoder
def labelling(X_train,X_test):
    # differentiate between categorical/numerical columns
    catcol=X_train.dtypes=='object'
    categoricalcol=catcol[catcol].index
    
    labelencoder=LabelEncoder()
    X_trainpostpro=X_train.copy()
    X_testpostpro=X_test.copy()
    
    for col in categoricalcol:
        X_trainpostpro[col]= labelencoder.fit_transform(X_train[col])
        X_testpostpro[col]=labelencoder.transform(X_test[col])
        
    return X_trainpostpro, X_testpostpro



# Read Data from Data files
data=pd.read_csv("/Users/pankajsemwal/Test Repository/melb_data.csv")

# Define the X (input/independent) and Y (output/dependent) variables 
X=data[["Rooms","Type","Distance","Bedroom2","Bathroom","Landsize","BuildingArea","YearBuilt","Lattitude","Longtitude"]]
Y=data.Price

# split the data into training and test data
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2)

# Impute the null values in the data
X_trainpostimp,X_testpostimp=Imputer(X_train,X_test)


# Label Encode the categorical columns
X_trainpostpro,X_testpostpro=labelling(X_trainpostimp,X_testpostimp)


# Decision Tree 
modelregressor=DecisionTreeRegressor(random_state=1)
modelregressor.fit(X_trainpostpro,Y_train)
Y_predict=modelregressor.predict(X_testpostpro)
modelrandomregressor=RandomForestRegressor(n_estimators=100,randome_state=0)
modelrandomregressor.fit(X_trainpostpro,Y_train)
Y_predict2=modelrandomregressor.predict(X_testpostpro)


# Model accuracy check

meansquareerror=mean_squared_error(Y_test,Y_predict)
meansquareerror2=mean_squared_error(Y_test,Y_predict2)
print(meansquareerror)
print(meansquareerror2)
