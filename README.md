The goal of the research is to create a machine learning model that can effectively predict the characteristic mean free path of a material given its properties.

First, run the bash file, generate_data.sh, to create the csv files necessary to run the algorithm. Then, enter RF_regression.py and go to the __main__ method to check options for how to run the algorithm. The current options include:

- Splitting data into training and testing sets by temperature, compound, or randomly
- Choosing to run either Random Forest regression or Gradient Boost Method regression
- Selecting which plots to view

Then, run RF_regression.py to see the results of the model.
