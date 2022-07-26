import pandas as pd
from pymatgen.ext.matproj import MPRester

'''
This script queries the Materials Project database to find properties of
given materials to be used in calculations. The properties are stored in
a csv file.
'''

pretty_formulas = ["AlAs", "AlN", "GaAs", "GaN", "GaP", "Ge", "InAs", "InP", "Si", "Sn"]

# creates a Pandas dataframe to store the material properties
MAPI_KEY = "CDUNwAFSK48P6XzvSni"
material_ids = ["mp-2172", "mp-661", "mp-2534", "mp-804", "mp-2490", "mp-32", "mp-20305", "mp-20351", "mp-149", "mp-117"]
mat_properties = pd.DataFrame()



# Performing the query from the Materials Project database for every material id listed above
with MPRester(MAPI_KEY) as mpr:
    for mat_id in material_ids:
        # Gets the thermal and electric properties of the formula
        data = mpr.query(criteria={"material_id": mat_id}, properties=["material_id", "pretty_formula", "e_above_hull", "formation_energy_per_atom", "band_gap"])

        # Gets the elastic properties of the formula
        for entry, value in mpr.query(criteria={"material_id": mat_id}, properties=["elasticity"])[0]["elasticity"].items():
            if type(value) != list:
                data[0][entry] = value
        
        # Combining the properties into one dataframe
        df = pd.DataFrame(data)
        frames = [mat_properties, df]
        mat_properties = pd.concat(frames)



# Saving the dataframe to a csv file
mat_properties.to_csv("material_properties/material_properties.csv")
