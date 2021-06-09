# Spark Internship
# Linear Regression

```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn

#reading the data
sample=pd.read_csv("http://bit.ly/w-data") # to read data  from url
sample.head()
sample.describe()

#importing linear model package and fitting the data
from sklearn import linear_model # imorting required library files linear model
regr= linear_model.LinearRegression() #intializing
y=sample[['Scores']]
x= sample[['Hours']]
regr.fit(x,y)
regr.coef_
regr.intercept_
r_sq = regr.score(x, y)
print('coefficient of determination:', r_sq)
predicted_values= regr.predict(x)
sample["predicted"]= predicted_values.round(2)
sample.head()
residuals= (y-predicted_values)
residuals

#Residuals plot against Hours
plt.scatter(sample['Hours'],sample['residuals'], color= "green")
plt.xlabel('residuals')
plt.ylabel('Scores')

#predicted and fitted values plot against Hours
plt.scatter(sample['Hours'],sample['Scores'], color= "blue")
plt.scatter(sample['Hours'],sample['predicted'], color= "red")
#plt.scatter(sample['Hours'],sample['residuals'], color= "green")
plt.xlabel(' Hours')
plt.ylabel('Scores')

#adding residauls to dataframe
sample['residuals']= residuals
sample.describe()


#correlation among obsevred, fitted, residuals and hours
sample.corr()

#predicting scores for 9.25 hours
y_pred = regr.intercept_ + regr.coef_ * 9.25
print('predicted response:', y_pred, sep='\n')

```

***Thank You***
