import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
rand = 300
ALL_COMPOUNDS = ["AlGaAs", "GaAlN", "InGaP", "SiSn", "SiGe", "InAlAs", "GeSn"]



# SPLITTING DATA INTO TRAINING AND TEST SETS



def query_by_compound(compounds, specific_temps = None):
    '''
    Given a list of compounds and a list of temperatures, produces the corresponding 
    parameters (X) and output (y) dataframes of those compounds at those temperatures

    If specific_temps is None, then the data is queried for all temperatures
    '''
    labelled_data = pd.read_csv('material_properties/formatted_data_with_formulas.csv')
    
    # Looping through the compounds provided to append to the output dataframe
    data_of_interest = pd.DataFrame()
    for compound in compounds:
        if specific_temps:
            to_add = labelled_data.loc[(labelled_data['formula'] == compound) & (labelled_data['temperature'].isin(specific_temps))]
        else:
            to_add = labelled_data.loc[(labelled_data['formula'] == compound)]
        data_of_interest = pd.concat([data_of_interest, to_add])

    data_of_interest['expected_mfp'] = np.log(data_of_interest['expected_mfp'])
    
    # Formatting the data into parameters and output and returning
    X = data_of_interest.drop('expected_mfp', axis = 1).drop('Unnamed: 0', axis = 1).drop('formula', axis = 1)
    y = data_of_interest['expected_mfp']
    return X, y



def split_data_randomly(test_frac = 0.3):
    '''
    Splits data by a random value using sklearn model selection
    without regard for certain compounds or certain temperatures being in the
    training or test sets
    '''
    # Setting up the data and expected output
    expected_mfp = pd.read_csv("material_properties/formatted_data.csv")
    expected_mfp['expected_mfp'] = np.log(expected_mfp['expected_mfp'])

    y = expected_mfp["expected_mfp"]
    X = expected_mfp.drop('expected_mfp', axis = 1).drop('Unnamed: 0', axis = 1)
    print(X.columns)

    # Performing the split using sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_frac)
    
    # The output is a tuple of the four groups of data
    return X_train, X_test, y_train, y_test



def split_data_by_compound(training_set_compounds, test_set_compounds): 
    '''
    Splits the data into training and test sets by compound, where
    the inputs are lists of strings of compounds
    '''
    # Creating the training and test data sets using a helper function
    X_train, y_train = query_by_compound(training_set_compounds)
    X_test, y_test = query_by_compound(test_set_compounds)

    # The output is a tuple of the four groups of data
    return X_train, X_test, y_train, y_test



def split_data_by_temperature(training_set_temps, test_set_temps): 
    '''
    Splits the data into training and test sets by temperature, where the 
    inputs are lists of ints of temperatures
    '''
    # Creating the training and test data sets using a helper function
    X_train, y_train = query_by_compound(compounds = ALL_COMPOUNDS, specific_temps = training_set_temps)
    X_test, y_test = query_by_compound(compounds = ALL_COMPOUNDS, specific_temps = test_set_temps)

    # The output is a tuple of the four groups of data
    return X_train, X_test, y_train, y_test



# PERFORMING REGRESSION



def gradient_boost_regression(X_train, X_test, y_train, y_test, verbose = True, quick = False):
    '''
    Performs gradient boost regression on the data by first using cross validation grid search
    and then performing the actual regression

    If quick is True, then the grid search is performed on a smaller set of parameters

    If verbose is True, then the grid search is printed to the console
    '''
    if quick:
        #rf_grid_params = {'n_estimators': [250], 'learning_rate': [0.15], 'max_depth': [5]}
        rf_grid_params = {'n_estimators': [350], 'learning_rate': [0.15], 'max_depth': [3]}
    else:
        rf_grid_params = {'n_estimators': [250,400,550], 'learning_rate': [0.05, 0.15, 0.25], 'max_depth': [1,2,3,4,5]}
    grid = GridSearchCV(estimator = GradientBoostingRegressor(), param_grid = rf_grid_params, refit = True, verbose = 2, cv = 5)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    if verbose:
        print('Train data R-squared:', np.round(grid.score(X_train, y_train),3))
        print('Test data R-squared:', np.round(grid.score(X_test, y_test),3))

    y_hat = grid.predict(X_test)
    df_predictions = pd.DataFrame({'Accepted':y_test, 'Predicted':y_hat, 'Error': abs(1 - y_hat / y_test) * 100})
    df_predictions.sort_values(by='Error', inplace=True)
    print(df_predictions)
    return grid



def random_forest_regression(X_train, X_test, y_train, y_test, verbose = True, quick = False):
    '''
    Performs random forest regression on the data by first using cross validation grid search
    and then performing the actual regression

    If quick is True, then the grid search is performed on a smaller set of parameters

    If verbose is True, then the grid search is printed to the console
    '''
    if quick:
        rf_grid_params = {'n_estimators': [150], 'max_features': ['sqrt'], 'max_depth': [15]}
    else:
        rf_grid_params = {'n_estimators': [100, 125, 150, 175, 200], 'max_features': ['sqrt', 'log2'], 'max_depth': [10,13,15,17,20]}
    grid = GridSearchCV(estimator = RandomForestRegressor(), param_grid = rf_grid_params, refit = True, verbose = 2, cv = 5)
    grid.fit(X_train, y_train)

    if verbose:
        print('Train data R-squared:', np.round(grid.score(X_train, y_train),3))
        print('Test data R-squared:', np.round(grid.score(X_test, y_test),3))

    y_hat = grid.predict(X_test)
    df_predictions = pd.DataFrame({'Accepted':y_test, 'Predicted':y_hat, 'Error': abs(1 - y_hat / y_test) * 100})
    df_predictions.sort_values(by='Error', inplace=True)
    print(df_predictions)
    return grid



# PLOTTING THE RESULTS



def plot_pred_v_acc(X_test, y_test, grid, save = True):
    '''
    Plots a graph of the predicted values vs the actual values of the test set, where
    the values shown are the log of the characteristic mean free path
    '''
    y_hat = grid.predict(X_test)
    print(type(y_test))

    # Plotting the y = x line for comparison
    plt.plot(np.linspace(-17, -13, 100), np.linspace(-17, -13, 100), '-')

    # Plotting highlighted values where the composition = 0.9 to see if the pattern is detected
    y_hat_highlighted = []
    y_test_highlighted = []
    for index, row in X_test.iterrows():
        if row['composition'] == .9:
            i = index
            i = i % 77
            y_hat_highlighted.append(y_hat[i])
            y_test_highlighted.append(list(y_test)[i])

    # Generating the scatter plots and setting the axes
    plt.scatter(y_test_highlighted, y_hat_highlighted, alpha = 1 )
    plt.scatter(y_test, y_hat, alpha = 0.3)
    plt.xlabel('Accepted')
    plt.ylabel('Predicted')
    plt.title('Predicted v Accepted Log of Lambda_0')

    if save:
        plt.savefig("predictions_RF_reg.png")
        plt.close()
    else:
        plt.show()



def plot_comparison_at_500K(grid, test_set_compounds):
    '''
    Plotting the accepted / predicted values at 500 K for SiGe, InAlAs, GeSn
    '''
    X, y = query_by_compound(test_set_compounds, specific_temps = [500])

    # Here, we exponentiate measured y because the algorithm automatically takes the log
    # when querying the data
    y_exponential = []
    for y_val in y:
        y_exponential.append(np.exp(y_val))
    print(y_exponential)


    # Creating lists for every compound separately makes it easier to plot because the
    # Pandas dataframe is not allowed to be plotted
    measured_SiGe = y_exponential[0:11]
    measured_InAlAs = y_exponential[11:22]
    measured_GeSn = y_exponential[22:]

    y_hat = grid.predict(X)

    # Here, we exponentiate y_hat because the algorithm predicts log lambda_0
    y_hat_exponential = []
    for y_hat_val in y_hat:
        y_hat_exponential.append(np.exp(y_hat_val))

    pred_SiGe = y_hat_exponential[0:11]
    pred_InAlAs = y_hat_exponential[11:22]
    pred_GeSn = y_hat_exponential[22:]

    plt.plot(np.linspace(0,1,11), measured_SiGe, label = 'SiGe, BTE', color = 'r', marker = "s", alpha=0.2)
    plt.plot(np.linspace(0,1,11), measured_InAlAs, label = 'InAlAs, BTE', color = 'b', marker = "s", alpha=0.2)
    plt.plot(np.linspace(0,1,11), measured_GeSn, label = 'GeSn, BTE', color = 'g', marker = "s", alpha=0.2)
    plt.plot(np.linspace(0,1,11), pred_SiGe, label = 'SiGe, GBM', color = 'r', marker = "o", alpha=1)
    plt.plot(np.linspace(0,1,11), pred_InAlAs, label = 'InAlAs, GBM', color = 'b', marker = "o", alpha=1)
    plt.plot(np.linspace(0,1,11), pred_GeSn, label = 'GeSn, GBM', color = 'g', marker = "o", alpha = 1)
    plt.legend()
    plt.title("Characteristic MFP versus Concentration")
    plt.xlabel("Concentration (x)")
    plt.ylabel("Characteristic MFP (\u03BCm)")
    plt.xticks(np.linspace(0,1,11)) 
    plt.yticks(np.linspace(0.1e-6, 1e-6, 10))
    plt.savefig("mfp_versus_concentration_gbm.png")



if __name__ == '__main__':
    training_set_compounds = ["AlGaAs", "GaAlN", "InGaP", "SiSn"]
    test_set_compounds = ["SiGe", "InAlAs", "GeSn"]

    training_set_temps = [200,300,500,600,700,800]
    test_set_temps = [400]

    '''
    Comment out two of the three following lines to perform random data split or data split by compound/temperature
    '''
    X_train, X_test, y_train, y_test = split_data_randomly()
    #X_train, X_test, y_train, y_test = split_data_by_compound(training_set_compounds, test_set_compounds)
    #X_train, X_test, y_train, y_test = split_data_by_temperature(training_set_temps, test_set_temps)

    ''' 
    Comment out one of the following lines to perform random forest or gradient boost regression
    '''
    #grid = random_forest_regression(X_train, X_test, y_train, y_test, quick = True)
    grid = gradient_boost_regression(X_train, X_test, y_train, y_test, quick = False)

    '''
    Create the desired plots by commenting out the ones you don't want
    '''
    plot_pred_v_acc(X_test, y_test, grid)
    #plot_comparison_at_500K(grid, test_set_compounds)