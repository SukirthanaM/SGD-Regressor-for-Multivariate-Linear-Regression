# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Sukirthana.M
RegisterNumber: 212224220112

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())

x = df.drop(columns=['AveOccup', 'HousingPrice'])
y = df[['AveOccup', 'HousingPrice']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

sgd = SGDRegressor (max_iter=1000, tol=1e-3)

multi_output_sgd=MultiOutputRegressor(sgd)

multi_output_sgd.fit(x_train,y_train)

y_pred = multi_output_sgd.predict(x_test)

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

print("\nPredictions: \n", y_pred[:5])
*/
```

## Output:
![Screenshot 2025-03-12 112440](https://github.com/user-attachments/assets/47f9f1e9-3aff-4d6f-a8a9-5fe41922cd71)
![Screenshot 2025-03-12 112502](https://github.com/user-attachments/assets/0763dcae-fa30-4c78-923c-e21f4b3c5bff)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
