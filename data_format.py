import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import numpy as np

'''
This script converts raw materials data into a new Pandas dataframe
such that it is organized by compound and includes characteristic
mean free path as a function of composition and temperature.
'''

# loads the csv file containing the material properties
mat_properties = pd.read_csv("material_properties/material_properties.csv")

# Variables to store information about the materials
compounds = ["AlGaAs", "GaAlN", "GeSn", "InAlAs", "InGaP", "SiGe", "SiSn"]
compound_parts = [["AlAs", "GaAs"], ["AlN", "GaN"], ["Ge", "Sn"], ["InAs", "AlAs"], ["GaP", "InP"], ["Si", "Ge"], ["Si", "Sn"]]
x = np.linspace(0, 1, 11) # [0: 0.1: 1]
temps = np.linspace(200, 800, 7) # [200: 100: 800]



# Stores the provided MFP data by looping through every compound
# MFP_data is a 3D array with dimensions consisting of [compound_index, temp_index, x_index]
MFP_data = []
for compound in compounds:
    # Reads through the data provided in the MFP file
    with open("mfp_data/" +compound + "_Lambda_o.out") as f:
        data = f.readlines()
        polished_data = []
        for line in data[2:]:
            polished_data.append(line.replace("\n", "").split(" ")[1:])
        MFP_data.append(polished_data)



# Creates a Pandas dataframe that stores the expected mfp given the properties of the compound parts, the temperature, and the composition
formatted_data = pd.DataFrame()
for compound_index, compound in enumerate(compounds):
    # sets up a Pandas dataframe that stores temperatures and compositions
    df = pd.DataFrame()
    df["formula"] = ""
    df["temperature"] = 0
    df["composition"] = 0
    df["expected_mfp"] = 0

    for temp_index in range(len(temps)):
        for x_index in range(len(x)):
            # creates entries for every combination of temperature and composition
            df2 = pd.DataFrame({"formula": compound, "temperature": temps[temp_index], "composition": x[x_index], "expected_mfp": float(MFP_data[compound_index][temp_index][x_index])}, index=[0])
            df = pd.concat([df, df2])

    # looping through each compound part in the composition to add to the dataframe
    for j in range(2):
        # Gets the properties of the compound part
        compound_part = mat_properties[mat_properties["pretty_formula"] == compound_parts[compound_index][j]]

        for col, data in compound_part.iteritems():
            if col != "Unnamed: 0.1" and col != "material_id" and col != "pretty_formula":
                df["mat" + str(j) + "_" + col] = data.values[0]
            
    # Adds the Pandas dataframe to the Pandas dataframe that contains the expected values for the characteristic MFP given the compound parts'
    # properties and also the temperature and value of x (ranges between 0 and 1 with a step of 0.05)
    formatted_data = pd.concat([formatted_data, df])



formatted_data.to_csv("material_properties/formatted_data_with_formulas.csv")
formatted_data = formatted_data.drop('formula', axis = 1)
formatted_data.to_csv("material_properties/formatted_data.csv")