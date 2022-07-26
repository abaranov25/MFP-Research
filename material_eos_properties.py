from mp_api import MPRester
import pandas as pd

MAPI_KEY = 'b7nKrOV8pUN6jMnxVA8eArf0vAkkwgZk'
material_ids = ["mp-2172", "mp-661", "mp-2534", "mp-804", "mp-2490", "mp-32", "mp-20305", "mp-20351", "mp-149", "mp-117"]
pretty_formulas = ["AlAs", "AlN", "GaAs", "GaN", "GaP", "Ge", "InAs", "InP", "Si", "Sn"]

mat_properties = pd.read_csv('material_properties/material_properties.csv')
mat_eos_properties = pd.DataFrame()
with MPRester(MAPI_KEY) as mpr:
    for i, mat in enumerate(material_ids):
        try:
            eos_doc = mpr.eos.get_data_by_id(mat)
            V0 = eos_doc.eos['vinet']['V0']
            C = eos_doc.eos['vinet']['C']
            E0 = eos_doc.eos['vinet']['E0']
            B = eos_doc.eos['vinet']['B']
            data = {'V0': V0, 'C': C, 'E0': E0, 'B': B}
            mat_eos_properties = pd.concat([mat_eos_properties, pd.DataFrame(data, index=[i])])
        except:
            print('failed for material ', mat)

mat_properties = pd.concat([mat_properties, mat_eos_properties], axis = 1)

mat_properties.to_csv('material_properties/material_properties.csv')