# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:01:15 2020

@author: vic
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
dataset=pd.read_csv("Salary_Data.csv") 
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
ans=regressor.predict(X_test)
# =============================================================================
# TRAINING SET
# =============================================================================
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs exp(training set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# =============================================================================
# TEST SET 
# =============================================================================
plt.scatter(X_test,ans,color='red')
plt.plot(X_train,regressor.predict(X_train))
plt.title('Salary vs exp(Test set)')
plt.xlabel('experience')
plt.ylabel('salary')
plt.show()

# =============================================================================
# SALARY FOR 12 YEARS EXPERIENCE AND VALUE OF SLOPE AND INTERCEPT
# =============================================================================
pred_value=regressor.predict([[12]])
print(regressor.coef_)
print(regressor.intercept_)