import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler




# Setting up the data and expected output
expected_mfp = pd.read_csv("formatted_data.csv")
expected_mfp['expected_mfp'] = np.log(expected_mfp['expected_mfp'])

scale = StandardScaler()
expected_mfp_sc = scale.fit_transform(expected_mfp)
expected_mfp_sc = pd.DataFrame(expected_mfp_sc, columns=expected_mfp.columns)

y = expected_mfp["expected_mfp"]
X = expected_mfp.drop('expected_mfp', axis = 1).drop('Unnamed: 0', axis = 1)


from sklearn.model_selection import train_test_split
#rand = np.random.randint(0,10000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)#, random_state = rand)

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
'''
svr_param_grid = {'C': [20000, 25000, 30000, 35000, 40000], 'gamma': [0.03, 0.035, 0.04, 0.045, 0.05], 'kernel': ['rbf']}#, 'linear']}
grid = GridSearchCV(estimator=SVR(), param_grid = svr_param_grid, refit = True, verbose=2, cv=5)
grid.fit(X_train, y_train)
print(grid.best_params_)
y_hat = grid.predict(X_test)


print('Train data R-squared:', np.round(grid.score(X_train, y_train),3))
print('Test data R-squared:', np.round(grid.score(X_test, y_test),3))

'''
from sklearn.svm import SVR
Rsqr_test = []
Rsqr_train = []
for i in range(1):
    print('Beginning trial ',i)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2)#, random_state = rand)
    SVM_regression = SVR(C=30000, gamma = 0.03)
    SVM_regression.fit(X_train, y_train)
    #predictions = pd.DataFrame({'y_test': y_test, 'y_hat': y_hat})
    #print(predictions)

    train_score =  np.round(SVM_regression.score(X_train, y_train),5)
    test_score =  np.round(SVM_regression.score(X_test, y_test),5)

    print('Train data R-squared:', train_score)

    print('Test data R-squared:', test_score)

    Rsqr_test.append(test_score)
    Rsqr_train.append(train_score)

import matplotlib.pyplot as plt
y_hat = SVM_regression.predict(X_test)
# Plotting the accepted / predicted value graph
plt.plot(np.linspace(-17, -13, 100), np.linspace(-17, -13, 100), '-')
plt.scatter(y_test, y_hat)
plt.xlabel('Accepted')
plt.ylabel('Predicted')
plt.savefig("predictions_SVM_reg.png")

print(np.mean(Rsqr_test))
print(np.mean(Rsqr_train))
#'''