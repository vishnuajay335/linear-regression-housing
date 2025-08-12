import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Housing.csv")
categorical_cols = ['mainroad', 'guestroom', 'basement',
'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
x=df.drop(columns=['price'])
y=df['price']
print(x.head())
print(y.head())

#Splting 80% to train and 20% to test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print("\nx_train shape:",x_train.shape)
print("x_test shape:",x_test.shape)
print("y_train shape:",y_train.shape)
print("y_test shape:",y_test.shape)

#Fit a Linear Regression model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)

#Evaluate model using MAE, MSE, R².
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("\nMAE",mae)
print("MSE",mse)
print("R2 Sore",r2)

#Plot regression line
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()

#interpreting coefficients
print("\nIntercept (b0):",model.intercept_)
print("\nCoefficient (b1):", model.coef_)

#Feature names with coefficients
for col, coef in zip(x.columns, model.coef_):
    print(f"\n{col}: {coef:.2f}")
