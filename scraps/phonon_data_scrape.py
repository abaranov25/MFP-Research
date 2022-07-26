import pickle
import matplotlib.pyplot as plt
import numpy as np


with open('si_phonon_dos.pkl', 'rb') as f:
    dos = pickle.load(f)
    
    print()#loaded_dict['qpoints'])







    
















'''
    # dict_keys(['@module', '@class', 'lattice_rec', 'qpoints', 'bands', 'labels_dict', 'eigendisplacements', 'structure', 'has_nac'])

    
    bands = loaded_dict['bands']
    print(len(bands))
    print(len(bands[3]))

    for key in loaded_dict.keys():
        print("key: ", key)
        if key != "eigendisplacements":
            print(loaded_dict[key])
        if key == "bands" or key == "qpoints":
            print(np.array(loaded_dict[key]).size)
    print(loaded_dict.keys())

    for i in range(len(bands)):

        plt.plot(list(range(len(bands[i]))), bands[i], color="black")
    
    plt.show()

with open('si_phonon_dos.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    print(loaded_dict.keys())
    # dict_keys(['@module', '@class', 'structure', 'frequencies', 'densities', 'pdos'])
    print(len(loaded_dict['frequencies']))
    print(len(loaded_dict['densities']))
    freq = loaded_dict['frequencies']
    dens = loaded_dict['densities']
    pdos = loaded_dict['pdos']
    print(pdos)

    x = freq[dens.index(max(dens))]
    y = max(dens)
    plt.plot(x,y, "ro")
    plt.plot(freq, dens)
    plt.plot(freq, pdos[1])
    plt.show()

    #phonon_info = mpr.get_phonon_ddb_by_material_id("mp-149")
    #phonon_info = mpr.get_phonon_dos_by_material_id("mp-149")
    #print(phonon_info.as_dict())
'''

#mat_properties.to_csv("material_properties.csv")