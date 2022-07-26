import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import matplotlib.pyplot as plt


# Setting up the data and expected output
expected_mfp = pd.read_csv("formatted_data.csv")
y = expected_mfp["expected_mfp"]
X = expected_mfp.drop('expected_mfp', axis = 1).drop('Unnamed: 0', axis = 1)


# Taking the logarithm of every lambda_0 for a better fit
import math
new_y = pd.DataFrame({"expected_mfp"})
for col_name, data in y.items():
    y[col_name] = math.log(data)


# Splitting the data into a training and test set
rand_state = 300
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=rand_state)


# NOT USING K-FOLD CROSS VALIDATION
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_hat = reg_model.predict(X_test)


# USING K-FOLD CROSS VALIDATION
#y_hat = cross_val_predict(reg_model, X, y, cv=10)
#MSE = cross_val_score()


# Creating a data frame to track error in predictions
df_predictions = pd.DataFrame({'Accepted':y_test, 'Predicted':y_hat, 'Error': abs(1 - y_hat / y_test) * 100})
df_predictions.sort_values(by='Error', inplace=True)
print(df_predictions)


reg_summary = pd.DataFrame(data = X_train.columns, columns=['Features'])
reg_summary['Coefficients'] = np.round(reg_model.coef_,4)
print(reg_summary)
print('Test data R-squared:', np.round(reg_model.score(X_test, y_test),3))


# Plotting the accepted / predicted value graph
plt.scatter(df_predictions['Accepted'], df_predictions['Predicted'])
plt.plot(np.linspace(-20, -10, 100), np.linspace(-20, -10, 100), '-')
plt.xlabel('Accepted')
plt.ylabel('Predicted')
plt.savefig("predictions_linear_reg.png")