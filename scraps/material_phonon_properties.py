import pandas as pd
from pymatgen.ext.matproj import MPRester 
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pymatgen.phonon.dos as dos

pretty_formulas = ["AlAs", "AlN", "GaAs", "GaN", "GaP", "Ge", "InAs", "InP", "Si", "Sn"]
material_ids = ["mp-2172", "mp-661", "mp-2534", "mp-804", "mp-2490", "mp-32", "mp-20305", "mp-20351", "mp-149", "mp-117"]
MAPI_KEY = "CDUNwAFSK48P6XzvSni"

mat_phonon_properties = pd.DataFrame()

temps = np.linspace(200, 800, 7) # [200: 100: 800]

'''

with MPRester(MAPI_KEY) as mpr:
    for mat in pretty_formulas:
        print(mpr.query({"pretty_formula":mat,"has": "bandstructure"},["material_id","pretty_formula"]))
        print(mpr.get_phonon_ddb_by_material_id("mp-32"))
'''
with MPRester(MAPI_KEY) as mpr:
    for mat in material_ids:
        try:
            print(mpr.query({"material_id":mat,"has": "bandstructure"},["material_id","pretty_formula"]))
            phonon_dos = mpr.get_phonon_dos_by_material_id(mat)
            #first_pos_freq = phonon_dos.ind_zero_freq
            free_energy = phonon_dos.helmholtz_free_energy(t=0)
            #entropy = phonon_dos.entropy(t=0)
            #internal_energy = phonon_dos.internal_energy(t=0)
            data = {'mat_id': mat, 'free_energy': free_energy}
            mat_phonon_properties = pd.concat([mat_phonon_properties, pd.DataFrame(data, index=[0])])
        except:
            print('failed for material ', mat)
mat_phonon_properties.to_csv('mat_phonon_properties.csv')



'''
    
'''
phonon_info = mpr.get_phonon_bandstructure_by_material_id("mp-32")
phonon_info_2 = mpr.get_phonon_ddb_by_material_id("mp-32")
phonon_info_3 = mpr.get_phonon_dos_by_material_id("mp-32")

with open('si_phonon_bandstructure.pkl', 'wb') as f:
    pickle.dump(phonon_info.as_dict(), f)

with open('si_phonon_ddb.txt', 'w') as f:
    f.write(phonon_info_2)

with open('si_phonon_dos.pkl', 'wb') as f:
    pickle.dump(phonon_info_3.as_dict(), f)
'''
'''
