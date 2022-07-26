import math
from random import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import rand
from sklearn.linear_model import LinearRegression

# Setting up the data and expected output
expected_mfp = pd.read_csv("formatted_data.csv")
y = np.array(expected_mfp["expected_mfp"])
X = expected_mfp.drop('expected_mfp', axis = 1).drop('Unnamed: 0', axis = 1)

# Normalizing all data for regularization to work
for i, data in enumerate(y):
    y[i] = math.log(data)
y = (y - y.mean()) / y.std()

for col_name, data in X.items():
    if data.std() != 0:
        X[col_name] = (data - data.mean()) / data.std()

from sklearn.model_selection import train_test_split
rand_state = 1000
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = rand_state)


from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, ElasticNet, ElasticNetCV

ridgecv = RidgeCV(alphas=list(10 ** np.linspace(-10,2,130)))
ridgecv.fit(X_train, y_train)
ridge_alpha_opt = ridgecv.alpha_

lassocv = LassoCV()
lassocv.fit(X_train, y_train)
lasso_alpha_opt = lassocv.alpha_

netcv = ElasticNetCV()
netcv.fit(X_train, y_train)
netcv_alpha_opt = netcv.alpha_

print("ridge alpha: ", ridge_alpha_opt)
print("lasso alpha: ", lasso_alpha_opt)
print("netcv alpha: ", netcv_alpha_opt)

model_linear = LinearRegression()
model_ridge = Ridge(alpha = ridge_alpha_opt)
model_lasso = Lasso(alpha = lasso_alpha_opt)
model_net = ElasticNet(alpha = netcv_alpha_opt)

y_hat_linear = model_linear.fit(X_train, y_train).predict(X_test)
y_hat_ridge = model_ridge.fit(X_train, y_train).predict(X_test)
y_hat_lasso = model_lasso.fit(X_train, y_train).predict(X_test)
y_hat_net = model_net.fit(X_train, y_train).predict(X_test)


print("The R^2 for Linear Regression is " + str(model_linear.score(X_test, y_test)))
print("The R^2 for Ridge Regression is " + str(model_ridge.score(X_test, y_test)))
print("The R^2 for Lasso Regression is " + str(model_lasso.score(X_test, y_test)))
print("The R^2 for ElasticNet Regression is " + str(model_net.score(X_test, y_test)))
model_predictions = {"y_test": y_test,
                                "y_hat_linear": y_hat_linear,
                                "y_hat_ridge": y_hat_ridge,
                                "y_hat_lasso": y_hat_lasso,
                                "y_hat_net": y_hat_net}

df_predictions = pd.DataFrame(model_predictions)

print(df_predictions)