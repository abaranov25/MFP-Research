import pandas as pd
from pyrsistent import s
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
for col_name, data in y.items():
    y[col_name] = math.log(data)

deg = 3

# Create the polynomial for X
poly_features = PolynomialFeatures(degree=deg)
new_X = pd.DataFrame()

for attribute_name, attribute_data in X.items():
    #print(attribute_name)
    #print(attribute_data[0])

    arr = []
    for attribute_data_point in attribute_data:
        arr.append(attribute_data_point)
    arr = np.array(arr)

    

    new_attribute_data = pd.DataFrame(poly_features.fit_transform(arr.reshape(-1,1)))

    new_attribute_data.columns = [(attribute_name + " ^ 0"), 
                                    (attribute_name + " ^ 1"), 
                                    (attribute_name + " ^ 2"), 
                                    (attribute_name + " ^ 3"),
                                    (attribute_name + " ^ 4"),
                                    (attribute_name + " ^ 5"),
                                    (attribute_name + " ^ 6"),
                                    (attribute_name + " ^ 7"),
                                    (attribute_name + " ^ 8"),
                                    (attribute_name + " ^ 9"),
                                    (attribute_name + " ^ 10")][0 : deg + 1]

    new_attribute_data = new_attribute_data.drop(attribute_name + " ^ 0", axis = 1)

    new_X = pd.concat([new_X, new_attribute_data], axis = 1)


# Split the data into train/test
rand_state = 300
X_train, X_test, y_train, y_test = train_test_split(new_X, y, random_state = rand_state)

# Create the regression for the training data
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

# Predict for test data
y_hat = reg_model.predict(X_test)




# Creating a data frame to track error in predictions
df_predictions = pd.DataFrame({'Accepted':y_test, 'Predicted':y_hat, 'Error': abs(1 - y_hat / y_test) * 100})
df_predictions.sort_values(by='Error', inplace=True)
print(df_predictions)


reg_summary = pd.DataFrame(data = X_train.columns, columns=['Features'])
reg_summary['Coefficients'] = np.round(reg_model.coef_,30)
reg_summary.sort_values(by='Coefficients', inplace = True)
print(reg_summary)
print('Test data R-squared:', np.round(reg_model.score(X_test, y_test),3))
#print(reg_model.coef_)


# Plotting the accepted / predicted value graph
plt.scatter(df_predictions['Accepted'], df_predictions['Predicted'])
plt.plot(np.linspace(-20, -10, 100), np.linspace(-20, -10, 100), '-')
plt.xlabel('Accepted')
plt.ylabel('Predicted')
plt.savefig("predictions_poly_reg.png")